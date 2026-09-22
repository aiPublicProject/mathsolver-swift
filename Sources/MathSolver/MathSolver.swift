import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// BYOK AI math solver with independent verification.
/// An answer is only `verified: true` when the model's verification
/// expression (pure arithmetic) is evaluated locally and matches.
public final class MathSolver {
    private let apiKey: String
    private let baseUrl: String
    private let model: String
    private let transport: Transport

    /// BYOK client for an OpenAI-compatible endpoint. Instantiate once, solve many.
    /// - Parameters:
    ///   - apiKey: user's own key (BYOK)
    ///   - baseUrl: any OpenAI-compatible endpoint, e.g. https://api.deepseek.com/v1
    ///   - model: model id; nil = gpt-4o-mini
    ///   - transport: test injection; nil = built-in HTTP
    public init(apiKey: String, baseUrl: String = "https://api.openai.com/v1",
                model: String? = nil, transport: Transport? = nil) throws {
        if apiKey.isEmpty { throw SolverError("NO_API_KEY", "apiKey is required (BYOK)") }
        let base = baseUrl.replacingOccurrences(of: "/+$", with: "", options: .regularExpression)
        if !(base.hasPrefix("http://") || base.hasPrefix("https://")) {
            throw SolverError("BAD_BASE_URL", "baseUrl must be an http(s) URL, e.g. https://api.deepseek.com/v1")
        }
        self.apiKey = apiKey
        self.baseUrl = base
        self.model = model ?? "gpt-4o-mini"
        self.transport = transport ?? MathSolver.defaultTransport
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
    {"answer": <number>, "steps": [<string>, ...], "verification": {"expression": "<string>"}}
    Rules:
    - "answer" must be a single number (the final result).
    - "steps" must be an array of short plain-language explanation strings.
    - "verification.expression" must be a pure arithmetic expression that
      evaluates to the answer. Allowed: numbers, + - * / % ^ ( ), and the
      functions abs sqrt sin cos tan ln log exp floor ceil round min max
      (log is base 10, ln is natural), and the constants pi and e.
    - The expression must recompute the answer independently.
    """

    public struct SolveResult {
        public let answer: Double
        public let steps: [String]
        public let expression: String
        public let evaluated: Double?
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

    static func numericallyEqual(_ a: Double, _ b: Double) -> Bool {
        Swift.abs(a - b) <= 1e-6 * Swift.max(1, Swift.max(Swift.abs(a), Swift.abs(b)))
    }

    // MARK: - JSON

    struct Parsed { let answer: Double; let steps: [String]; let expression: String }

    static func parseModelReply(_ text: String) throws -> Parsed {
        guard let start = text.firstIndex(of: "{"), let end = text.lastIndex(of: "}"), start < end else {
            throw SolverError("INVALID_JSON", "no JSON object in reply")
        }
        let body = String(text[start...end])
        guard let data = body.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SolverError("INVALID_JSON", "reply was not valid JSON")
        }
        var answer: Double?
        if let n = obj["answer"] as? Double { answer = n }
        else if let n = obj["answer"] as? Int { answer = Double(n) }
        else if let s = obj["answer"] as? String,
                let m = s.range(of: #"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"#, options: .regularExpression) {
            answer = Double(s[m])
        }
        guard let finalAnswer = answer else { throw SolverError("INVALID_JSON", "missing numeric answer") }
        guard let verification = obj["verification"] as? [String: Any],
              let expression = verification["expression"] as? String else {
            throw SolverError("INVALID_JSON", "missing verification.expression")
        }
        let steps = (obj["steps"] as? [Any])?.compactMap { $0 as? String } ?? []
        return Parsed(answer: finalAnswer, steps: steps, expression: expression)
    }

    // MARK: - solve

    public func solve(_ problem: String) throws -> SolveResult {
        if problem.trimmingCharacters(in: .whitespaces).isEmpty {
            throw SolverError("NO_PROBLEM", "problem must be non-empty")
        }
        let tr = transport
        let url = baseUrl + "/chat/completions"
        var messages: [[String: String]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": problem]
        ]
        func call() throws -> String {
            let body = try! JSONSerialization.data(withJSONObject: ["model": self.model, "messages": messages, "temperature": 0])
            return try tr(url, String(data: body, encoding: .utf8)!, self.apiKey)
        }

        var parsed: Parsed
        do {
            parsed = try parseModelReply(try call())
        } catch let e as SolverError where e.code == "INVALID_JSON" {
            messages.append(["role": "assistant", "content": "invalid JSON"])
            messages.append(["role": "user", "content": "Your reply was not valid JSON. Reply again with the exact strict JSON shape."])
            parsed = try parseModelReply(try call())
        }

        func evaluate(_ p: Parsed) -> (Double?, Bool) {
            guard let ev = try? evalExpression(p.expression) else { return (nil, false) }
            return (ev, numericallyEqual(ev, p.answer))
        }

        var (evaluated, verified) = evaluate(parsed)
        var retries = 0
        if !verified {
            retries = 1
            messages.append(["role": "user", "content":
                "Your verification expression evaluated to \(evaluated.map(String.init) ?? "an error"), " +
                "which does not match your answer \(parsed.answer). " +
                "Re-derive carefully and reply again with the same strict JSON shape."])
            if let second = try? parseModelReply(try call()) {
                let (ev2, ok2) = evaluate(second)
                if let ev2 = ev2 { evaluated = ev2 }
                if ok2 { parsed = second; verified = true }
            }
        }
        return SolveResult(answer: parsed.answer, steps: parsed.steps, expression: parsed.expression,
                           evaluated: evaluated, verified: verified, retries: retries)
    }

    public static let defaultTransport: Transport = { url, body, apiKey in
        var request = URLRequest(url: URL(string: url)!)
        request.httpMethod = "POST"
        request.httpBody = body.data(using: .utf8)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        let (data, response) = try URLSession.shared.synchronous(request)
        guard let http = response as? HTTPURLResponse else { throw SolverError("HTTP_ERROR", "no response") }
        if http.statusCode >= 300 { throw SolverError("HTTP_ERROR", "API responded \(http.statusCode)") }
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = obj["choices"] as? [[String: Any]], let first = choices.first,
              let message = first["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw SolverError("HTTP_ERROR", "missing message content")
        }
        return content
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
