import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// BYOK AI math solver with execution-based verification (v0.2).
///
/// Correctness model (PAL-style): the model never states the answer.
/// It returns a small JavaScript-like PROGRAM; this library executes the
/// program deterministically and the execution output IS the answer.
/// For equations, a CHECK expression ({x} placeholder) must evaluate to 0
/// when the computed answer is substituted back into the original equation.
public final class MathSolver {
    private let apiKey: String
    private let baseUrl: String
    private let model: String
    private let transport: Transport

    /// Test seam for the HTTP interface below the default transport:
    /// (url, headers, bodyJson) -> (status, raw body). Swap it in tests to run
    /// the real default-transport code path without sockets.
    public typealias HttpPost = (_ url: String, _ headers: [String: String], _ body: String) throws -> (Int, String)

    /// BYOK client for an OpenAI-compatible endpoint. Instantiate once, solve many.
    /// - Parameters:
    ///   - apiKey: user's own key (BYOK)
    ///   - baseUrl: any OpenAI-compatible endpoint, e.g. https://api.deepseek.com/v1
    ///   - model: model id; nil = gpt-4o-mini
    ///   - transport: test injection; nil = built-in HTTP
    ///   - httpPost: HTTP-layer test seam; nil = real URLSession POST
    public init(apiKey: String, baseUrl: String = "https://api.openai.com/v1",
                model: String? = nil, transport: Transport? = nil, httpPost: HttpPost? = nil) throws {
        if apiKey.isEmpty { throw SolverError("NO_API_KEY", "apiKey is required (BYOK)") }
        let base = baseUrl.replacingOccurrences(of: "/+$", with: "", options: .regularExpression)
        if !(base.hasPrefix("http://") || base.hasPrefix("https://")) {
            throw SolverError("BAD_BASE_URL", "baseUrl must be an http(s) URL, e.g. https://api.deepseek.com/v1")
        }
        self.apiKey = apiKey
        self.baseUrl = base
        self.model = model ?? "gpt-4o-mini"
        let post = httpPost ?? MathSolver.realHTTPPost
        self.transport = transport ?? { url, body, key in
            let (status, raw) = try post(url, ["Content-Type": "application/json", "Authorization": "Bearer \(key)"], body)
            return try MathSolver.contentFromResponse(status: status, raw: raw)
        }
    }

    public struct SolverError: Error, CustomStringConvertible {
        public let code: String
        public let message: String
        public init(_ code: String, _ message: String) { self.code = code; self.message = message }
        public var description: String { "\(code): \(message)" }
    }

    public static let systemPrompt = """
    You are a precise math solver.
    Reply with STRICT JSON only, no markdown fences, in this exact shape:
    {"program": "<string>", "steps": [<string>, ...], "check": "<string>"}
    Rules:
    - "program" is a small JavaScript-like program that computes the final answer.
      One statement per line (or ; separated). Allowed statements:
          let NAME = EXPRESSION
          result = EXPRESSION
      EXPRESSIONs may use numbers, + - * / % ^ ( ), the functions
      abs sqrt sin cos tan ln log exp floor ceil round min max
      (log is base 10, ln is natural), the constants pi and e, and any
      variable defined by an earlier let. The value assigned to "result"
      is the answer. Never state the answer as a number in text.
    - "steps" is an array of short plain-language explanation strings.
    - "check" is a verification expression containing the placeholder {x}.
      After solving, {x} is replaced by the computed answer and the whole
      expression must evaluate to 0.
      For equations, substitute the answer back into the original equation
      (e.g. 2x+3=11 -> "2*{x}+3-11").
      For arithmetic, recompute via a different path and subtract the answer
      (e.g. 15% of 80 -> "80*15/100-{x}"). Provide "check" whenever possible.
    """

    static func correctionPrompt(_ reason: String) -> String {
        "Your submission failed verification: \(reason). " +
        "Re-derive the problem carefully and reply again with the same strict JSON shape."
    }

    /// answer is the output of executing the model's program locally;
    /// verified is true only when the check expression ({x} substituted with
    /// the answer) evaluated to ~0. check/checkValue are nil when none provided.
    public struct SolveResult {
        public let answer: Double
        public let steps: [String]
        public let program: String
        public let check: String?
        public let checkValue: Double?
        public let verified: Bool
        public let retries: Int
    }

    public typealias Transport = (_ url: String, _ body: String, _ apiKey: String) throws -> String

    // MARK: - Expression evaluator

    private struct Tok {
        enum Kind { case num(Double); case id(String); case op(String) }
        let kind: Kind
    }

    private static func tokenize(_ src: String) throws -> [Tok] {
        var tokens: [Tok] = []
        let chars = Array(src)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c == " " || c == "\t" || c == "\n" || c == "\r" { i += 1; continue }
            if c.isNumber || c == "." {
                var j = i
                while j < chars.count && (chars[j].isNumber || chars[j] == ".") { j += 1 }
                if j < chars.count, "eE".contains(chars[j]), j + 1 < chars.count, chars[j + 1].isNumber || "+-".contains(chars[j + 1]) {
                    j += 2
                    while j < chars.count && chars[j].isNumber { j += 1 }
                }
                guard let v = Double(String(chars[i..<j])) else {
                    throw SolverError("EXPR_BAD_NUMBER", "bad number")
                }
                tokens.append(Tok(kind: .num(v)))
                i = j
                continue
            }
            if c.isLetter || c == "_" {
                var j = i
                while j < chars.count && (chars[j].isLetter || chars[j].isNumber || chars[j] == "_") { j += 1 }
                tokens.append(Tok(kind: .id(String(chars[i..<j]))))
                i = j
                continue
            }
            if "+-*/%^(),".contains(c) {
                tokens.append(Tok(kind: .op(String(c))))
                i += 1
                continue
            }
            throw SolverError("EXPR_BAD_CHAR", "unexpected character \(c)")
        }
        return tokens
    }

    /// Evaluate a pure arithmetic expression string.
    public static func evalExpression(_ src: String) throws -> Double {
        try evalExpression(src, env: [:])
    }

    /// Evaluate with variable bindings; names are case-sensitive and shadow pi/e.
    public static func evalExpression(_ src: String, env: [String: Double]) throws -> Double {
        if src.trimmingCharacters(in: .whitespaces).isEmpty {
            throw SolverError("EXPR_EMPTY", "empty expression")
        }
        let tokens = try tokenize(src)
        var pos = 0

        func peek() -> Tok? { pos < tokens.count ? tokens[pos] : nil }
        @discardableResult func eat() throws -> Tok {
            guard pos < tokens.count else { throw SolverError("EXPR_SYNTAX", "expected more tokens") }
            defer { pos += 1 }
            return tokens[pos]
        }

        func expr() throws -> Double {
            var v = try term()
            while let t = peek(), case .op(let op) = t.kind, op == "+" || op == "-" {
                _ = try eat()
                let r = try term()
                v = op == "+" ? v + r : v - r
            }
            return v
        }

        func term() throws -> Double {
            var v = try unary()
            while let t = peek(), case .op(let op) = t.kind, ["*", "/", "%"].contains(op) {
                _ = try eat()
                let r = try unary()
                switch op {
                case "*": v = v * r
                case "/": v = v / r
                default: v = v.truncatingRemainder(dividingBy: r)
                }
            }
            return v
        }

        func unary() throws -> Double {
            if let t = peek(), case .op("-") = t.kind { _ = try eat(); return try -unary() }
            if let t = peek(), case .op("+") = t.kind { _ = try eat(); return try unary() }
            return try power()
        }

        func power() throws -> Double {
            let base = try atom()
            if let t = peek(), case .op("^") = t.kind {
                _ = try eat()
                return Foundation.pow(base, try unary()) // right associative
            }
            return base
        }

        func atom() throws -> Double {
            let t = try eat()
            switch t.kind {
            case .num(let v): return v
            case .id(let raw):
                if let bound = env[raw] { return bound }
                let name = raw.lowercased()
                if let n = peek(), case .op("(") = n.kind {
                    _ = try eat()
                    var args = [try expr()]
                    while let n = peek(), case .op(",") = n.kind {
                        _ = try eat()
                        args.append(try expr())
                    }
                    guard case .op(")") = (try eat()).kind else { throw SolverError("EXPR_SYNTAX", "expected )") }
                    return try applyFn(name, args)
                }
                if name == "pi" { return Double.pi }
                if name == "e" { return M_E }
                throw SolverError("EXPR_UNKNOWN_ID", "unknown identifier \(name)")
            case .op("("):
                let v = try expr()
                guard case .op(")") = (try eat()).kind else { throw SolverError("EXPR_SYNTAX", "expected )") }
                return v
            case .op(let other):
                throw SolverError("EXPR_SYNTAX", "unexpected token \(other)")
            }
        }

        func applyFn(_ name: String, _ args: [Double]) throws -> Double {
            let a0 = args.first ?? .nan
            switch name {
            case "abs": return Swift.abs(a0)
            case "sqrt": return Foundation.sqrt(a0)
            case "sin": return Foundation.sin(a0)
            case "cos": return Foundation.cos(a0)
            case "tan": return Foundation.tan(a0)
            case "ln": return Foundation.log(a0)
            case "log": return Foundation.log10(a0)
            case "exp": return Foundation.exp(a0)
            case "floor": return Foundation.floor(a0)
            case "ceil": return Foundation.ceil(a0)
            case "round": return Foundation.round(a0)
            case "min": return args.min() ?? .nan
            case "max": return args.max() ?? .nan
            default: throw SolverError("EXPR_UNKNOWN_FUNC", "unknown function \(name)")
            }
        }

        let value = try expr()
        if pos != tokens.count { throw SolverError("EXPR_TRAILING", "trailing tokens") }
        if value.isNaN || value.isInfinite { throw SolverError("EXPR_NON_FINITE", "non-finite result") }
        return value
    }

    // MARK: - Program interpreter

    /// Execute a model-generated program. Statements (one per line or ;
    /// separated): let NAME = EXPR | NAME = EXPR | bare EXPR. The answer is
    /// the value of `result`, else the last bare expression. The model never
    /// states the answer as a number — execution output IS the answer.
    public static func runProgram(_ src: String) throws -> Double {
        if src.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw SolverError("PROGRAM_EMPTY", "empty program")
        }
        var env: [String: Double] = [:]
        var resultDefined = false
        var lastDefined = false
        var lastValue = 0.0
        for raw in src.components(separatedBy: CharacterSet(charactersIn: ";\n")) {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }
            if let (name, rhs) = matchAssignment(line) {
                env[name] = try evalExpression(rhs, env: env)
                if name == "result" { resultDefined = true }
                continue
            }
            lastValue = try evalExpression(line, env: env)
            lastDefined = true
        }
        if resultDefined { return env["result"]! }
        if lastDefined { return lastValue }
        throw SolverError("PROGRAM_NO_RESULT", "program produced no result")
    }

    /// Matches "let NAME = RHS" or "NAME = RHS"; nil when the line is a bare expression.
    private static func matchAssignment(_ line: String) -> (String, String)? {
        var s = Substring(line)
        if s.hasPrefix("let") {
            let after = s.dropFirst(3)
            guard let first = after.first, first == " " || first == "\t" else { return nil }
            s = after.drop { $0 == " " || $0 == "\t" }
        }
        guard let eq = s.firstIndex(of: "=") else { return nil }
        let name = String(s[..<eq]).trimmingCharacters(in: .whitespaces)
        guard isIdentifier(name) else { return nil }
        let rhs = String(s[s.index(after: eq)...]).trimmingCharacters(in: .whitespaces)
        guard !rhs.isEmpty else { return nil }
        return (name, rhs)
    }

    private static func isIdentifier(_ s: String) -> Bool {
        guard let first = s.first, first.isLetter || first == "_" else { return false }
        return s.dropFirst().allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
    }

    /// Substitute the computed answer into a check expression ({x} placeholder)
    /// and evaluate it. Passes when the value is ~0 (scaled tolerance).
    public static func runCheck(_ checkSrc: String, answer: Double) throws -> (value: Double, passed: Bool) {
        let substituted = checkSrc.replacingOccurrences(
            of: #"\{\s*x\s*\}"#,
            with: "(\(answer))",
            options: [.regularExpression, .caseInsensitive]
        )
        let value = try evalExpression(substituted)
        return (value, Swift.abs(value) <= 1e-6 * Swift.max(1, Swift.abs(answer)))
    }

    // MARK: - JSON

    struct Parsed { let program: String; let steps: [String]; let check: String? }

    static func parseModelReply(_ text: String) throws -> Parsed {
        guard let start = text.firstIndex(of: "{"), let end = text.lastIndex(of: "}"), start < end else {
            throw SolverError("INVALID_JSON", "no JSON object in reply")
        }
        let body = String(text[start...end])
        guard let data = body.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SolverError("INVALID_JSON", "reply was not valid JSON")
        }
        guard let program = obj["program"] as? String,
              !program.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SolverError("INVALID_JSON", "missing program")
        }
        let steps = (obj["steps"] as? [Any])?.compactMap { $0 as? String } ?? []
        var check: String?
        if let c = obj["check"] as? String, !c.trimmingCharacters(in: .whitespaces).isEmpty {
            check = c
        }
        return Parsed(program: program, steps: steps, check: check)
    }

    /// Re-serialize a parsed reply for the corrective retry message.
    static func parsedJSON(_ p: Parsed) -> String {
        let stepsJSON = p.steps.map { "\"\($0.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\""))\"" }
            .joined(separator: ",")
        let checkJSON = p.check.map { "\"\($0.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\""))\"" } ?? "null"
        let programJSON = p.program.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        return "{\"program\": \"\(programJSON)\", \"steps\": [\(stepsJSON)], \"check\": \(checkJSON)}"
    }

    // MARK: - solve

    private struct Attempt {
        var ok = false
        var answer = 0.0
        var checkValue: Double?
        var verified = false
        var err: SolverError?
    }

    private func attempt(_ p: Parsed) -> Attempt {
        var out = Attempt()
        do {
            out.answer = try MathSolver.runProgram(p.program)
            out.ok = true
            if let check = p.check {
                let r = try MathSolver.runCheck(check, answer: out.answer)
                out.checkValue = r.value
                out.verified = r.passed
            }
        } catch let e as SolverError {
            out.err = e
        } catch {
            out.err = SolverError("PROGRAM_ERROR", "\(error)")
        }
        return out
    }

    /// Solve a math problem. `answer` is the output of executing the model's
    /// program; `verified` is true only when the check expression ({x}
    /// substituted with the answer) evaluated to ~0.
    public func solve(_ problem: String) throws -> SolveResult {
        if problem.trimmingCharacters(in: .whitespaces).isEmpty {
            throw SolverError("NO_PROBLEM", "problem must be non-empty")
        }
        let tr = transport
        let url = baseUrl + "/chat/completions"
        var messages: [[String: String]] = [
            ["role": "system", "content": MathSolver.systemPrompt],
            ["role": "user", "content": problem]
        ]
        func call() throws -> String {
            let body = try! JSONSerialization.data(withJSONObject: ["model": self.model, "messages": messages, "temperature": 0])
            return try tr(url, String(data: body, encoding: .utf8)!, self.apiKey)
        }

        var parsed: Parsed
        do {
            parsed = try MathSolver.parseModelReply(try call())
        } catch let e as SolverError where e.code == "INVALID_JSON" {
            messages.append(["role": "assistant", "content": "invalid JSON"])
            messages.append(["role": "user", "content": "Your reply was not valid JSON. Reply again with the exact strict JSON shape."])
            parsed = try MathSolver.parseModelReply(try call())
        }

        var outcome = attempt(parsed)
        var retries = 0
        if !outcome.ok || !outcome.verified {
            retries = 1
            let reason = outcome.ok
                ? "check evaluated to \(outcome.checkValue.map { String($0) } ?? "nil") instead of 0"
                : "program failed to execute (\(outcome.err?.code ?? "ERROR"): \(outcome.err?.message ?? ""))"
            messages.append(["role": "assistant", "content": MathSolver.parsedJSON(parsed)])
            messages.append(["role": "user", "content": MathSolver.correctionPrompt(reason)])
            let secondParsed = try MathSolver.parseModelReply(try call()) // second failure throws
            let second = attempt(secondParsed)
            if !second.ok { throw second.err ?? SolverError("PROGRAM_ERROR", "unknown") } // PROGRAM_* persisted after retry
            parsed = secondParsed
            outcome = second
        }

        return SolveResult(answer: outcome.answer, steps: parsed.steps, program: parsed.program,
                           check: parsed.check, checkValue: outcome.checkValue,
                           verified: outcome.verified, retries: retries)
    }

    // MARK: - HTTP interface

    /// Real HTTP POST via URLSession. Returns (status, raw body).
    public static let realHTTPPost: HttpPost = { url, headers, body in
        var request = URLRequest(url: URL(string: url)!)
        request.httpMethod = "POST"
        request.httpBody = body.data(using: .utf8)
        for (name, value) in headers {
            request.setValue(value, forHTTPHeaderField: name)
        }
        let (data, response) = try URLSession.shared.synchronous(request)
        guard let http = response as? HTTPURLResponse else { throw SolverError("HTTP_ERROR", "no response") }
        return (http.statusCode, String(data: data, encoding: .utf8) ?? "")
    }

    static func contentFromResponse(status: Int, raw: String) throws -> String {
        if status >= 300 { throw SolverError("HTTP_ERROR", "API responded \(status)") }
        guard let data = raw.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = obj["choices"] as? [[String: Any]], let first = choices.first,
              let message = first["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw SolverError("HTTP_ERROR", "missing message content")
        }
        return content
    }

    public static let defaultTransport: Transport = { url, body, apiKey in
        let (status, raw) = try MathSolver.realHTTPPost(
            url,
            ["Content-Type": "application/json", "Authorization": "Bearer \(apiKey)"],
            body
        )
        return try MathSolver.contentFromResponse(status: status, raw: raw)
    }
}

private extension URLSession {
    func synchronous(_ request: URLRequest) throws -> (Data, URLResponse) {
        var result: Result<(Data, URLResponse), Error> = .failure(MathSolver.SolverError("HTTP_ERROR", "no response"))
        let sem = DispatchSemaphore(value: 0)
        let task = self.dataTask(with: request) { data, response, error in
            if let error = error { result = .failure(error) }
            else if let data = data, let response = response { result = .success((data, response)) }
            sem.signal()
        }
        task.resume()
        sem.wait()
        switch result {
        case .success(let value): return value
        case .failure(let error): throw error
        }
    }
}
