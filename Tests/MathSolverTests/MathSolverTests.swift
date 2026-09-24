import XCTest
@testable import MathSolver

final class MathSolverTests: XCTestCase {
    // v0.2 protocol fixtures: the model returns program/steps/check — never an answer.
    let good = #"{"program": "let d = 11 - 3;\nlet x = d / 2;\nresult = x", "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"], "check": "2*{x} + 3 - 11"}"#
    let noCheck = #"{"program": "result = 0.15 * 80", "steps": ["Compute 15% of 80"]}"#
    let wrongCheck = #"{"program": "let d = 11 - 3;\nresult = d / 2", "steps": ["..."], "check": "2*{x} + 3 - 12"}"#
    let brokenProgram = #"{"program": "result = undefinedvar + 1", "steps": []}"#

    func code(_ error: Error) -> String? { (error as? MathSolver.SolverError)?.code }

    func testEvaluatorPrecedence() throws {
        XCTAssertEqual(try MathSolver.evalExpression("2*3+4"), 10, accuracy: 1e-9)
        XCTAssertEqual(try MathSolver.evalExpression("2+3*4"), 14, accuracy: 1e-9)
        XCTAssertEqual(try MathSolver.evalExpression("(2+3)*4"), 20, accuracy: 1e-9)
        XCTAssertEqual(try MathSolver.evalExpression("2^3^2"), 512, accuracy: 1e-9)
        XCTAssertEqual(try MathSolver.evalExpression("-3^2"), -9, accuracy: 1e-9)
    }

    func testEvaluatorFunctions() throws {
        XCTAssertEqual(try MathSolver.evalExpression("sqrt(16)"), 4, accuracy: 1e-9)
        XCTAssertEqual(try MathSolver.evalExpression("min(3,5)"), 3, accuracy: 1e-9)
        XCTAssertEqual(try MathSolver.evalExpression("pi"), Double.pi, accuracy: 1e-12)
        XCTAssertEqual(try MathSolver.evalExpression("log(1000)"), 3, accuracy: 1e-9)
    }

    func testEvaluatorEnvVariables() throws {
        XCTAssertEqual(try MathSolver.evalExpression("d / 2", env: ["d": 8]), 4, accuracy: 1e-9)
        XCTAssertEqual(try MathSolver.evalExpression("x + y", env: ["x": 1.5, "y": 2.5]), 4, accuracy: 1e-9)
        XCTAssertThrowsError(try MathSolver.evalExpression("d")) // undefined without env
        XCTAssertEqual(try MathSolver.evalExpression("pi", env: ["pi": 3]), 3, accuracy: 1e-12) // env shadows constant
    }

    func testEvaluatorRejectsBadInput() {
        for bad in ["Process.launch()", "1+2)", "foo(1)", ""] {
            XCTAssertThrowsError(try MathSolver.evalExpression(bad))
        }
    }

    func testRunProgram() throws {
        XCTAssertEqual(try MathSolver.runProgram("let d = 11 - 3;\nlet x = d / 2;\nresult = x"), 4, accuracy: 1e-9)
        XCTAssertEqual(try MathSolver.runProgram("let a = 3; let b = 4; a * b"), 12, accuracy: 1e-9)
        XCTAssertEqual(try MathSolver.runProgram("0.15 * 80"), 12, accuracy: 1e-9)
        for bad in ["result = undefinedvar + 1", "", "let a = 1; let b = 2"] {
            XCTAssertThrowsError(try MathSolver.runProgram(bad))
        }
    }

    func testRunCheck() throws {
        let pass = try MathSolver.runCheck("2*{x} + 3 - 11", answer: 4)
        XCTAssertTrue(pass.passed)
        XCTAssertEqual(pass.value, 0, accuracy: 1e-9)
        let fail = try MathSolver.runCheck("2*{x} + 3 - 12", answer: 4)
        XCTAssertFalse(fail.passed)
        XCTAssertEqual(fail.value, -1, accuracy: 1e-9)
        XCTAssertTrue(try MathSolver.runCheck("80*15/100 - {x}", answer: 12).passed)
    }

    func testInitValidatesCredentials() throws {
        XCTAssertThrowsError(try MathSolver(apiKey: "")) { e in
            XCTAssertEqual((e as? MathSolver.SolverError)?.code, "NO_API_KEY")
        }
        XCTAssertThrowsError(try MathSolver(apiKey: "sk", baseUrl: "not-a-url")) { e in
            XCTAssertEqual((e as? MathSolver.SolverError)?.code, "BAD_BASE_URL")
        }
    }

    func testSolveAnswerFromExecutionFirstTry() throws {
        var calls = 0
        var seenURL: String?, seenKey: String?, seenBody: String?
        let tr: MathSolver.Transport = { url, body, key in
            calls += 1; seenURL = url; seenKey = key; seenBody = body
            return self.good
        }
        let solver = try MathSolver(apiKey: "sk-test", baseUrl: "https://api.deepseek.com/v1", model: "deepseek-chat", transport: tr)
        let r = try solver.solve("2x + 3 = 11, solve for x")
        // 答案=执行产物(4), 代回检验=0; 模型 JSON 里没有 answer 字段
        XCTAssertTrue(r.verified)
        XCTAssertEqual(r.retries, 0)
        XCTAssertEqual(r.answer, 4, accuracy: 1e-9)
        XCTAssertEqual(r.checkValue ?? -1, 0, accuracy: 1e-9)
        XCTAssertEqual(calls, 1)
        XCTAssertEqual(seenURL, "https://api.deepseek.com/v1/chat/completions")
        XCTAssertEqual(seenKey, "sk-test")
        XCTAssertTrue(seenBody?.contains(#""model":"deepseek-chat""#) ?? false)
        XCTAssertTrue(seenBody?.contains(#""temperature":0"#) ?? false)
        XCTAssertFalse(good.contains(#""answer""#))
    }

    func testSolveNoCheckUnverified() throws {
        let tr: MathSolver.Transport = { _, _, _ in self.noCheck }
        let r = try MathSolver(apiKey: "sk", transport: tr).solve("15% of 80")
        XCTAssertEqual(r.answer, 12, accuracy: 1e-9)
        XCTAssertFalse(r.verified)
        XCTAssertNil(r.check)
        XCTAssertNil(r.checkValue)
    }

    func testSolveCheckFailRetryRecovers() throws {
        var n = 0
        let tr: MathSolver.Transport = { _, _, _ in
            n += 1
            return n == 1 ? self.wrongCheck : self.good
        }
        let r = try MathSolver(apiKey: "sk", transport: tr).solve("2x+3=11")
        XCTAssertTrue(r.verified)
        XCTAssertEqual(r.retries, 1)
        XCTAssertEqual(r.answer, 4, accuracy: 1e-9)
    }

    func testSolveProgramErrorRetryRecovers() throws {
        var n = 0
        let tr: MathSolver.Transport = { _, _, _ in
            n += 1
            return n == 1 ? self.brokenProgram : self.good
        }
        let r = try MathSolver(apiKey: "sk", transport: tr).solve("2x+3=11")
        XCTAssertTrue(r.verified)
        XCTAssertEqual(r.answer, 4, accuracy: 1e-9)
    }

    func testSolveProgramErrorPersists() {
        let tr: MathSolver.Transport = { _, _, _ in self.brokenProgram }
        XCTAssertThrowsError(try MathSolver(apiKey: "sk", transport: tr).solve("2x+3=11")) { e in
            let c = self.code(e) ?? ""
            XCTAssertTrue(c.hasPrefix("PROGRAM_") || c.hasPrefix("EXPR_"), "got \(c)")
        }
    }

    func testInvalidJSONThenOK() throws {
        var n = 0
        let tr: MathSolver.Transport = { _, _, _ in
            n += 1
            return n == 1 ? "no json" : self.good
        }
        let r = try MathSolver(apiKey: "sk", transport: tr).solve("1+1")
        XCTAssertTrue(r.verified)
    }

    func testInvalidTwiceRaises() {
        XCTAssertThrowsError(try MathSolver(apiKey: "sk", transport: { _, _, _ in "nothing" }).solve("1+1")) { error in
            XCTAssertEqual((error as? MathSolver.SolverError)?.code, "INVALID_JSON")
        }
    }

    func testNoAPIKeyCoveredAtInit() {
        XCTAssertTrue(true) // covered in testInitValidatesCredentials
    }

    func testHTTPErrorNoRetry() {
        var calls = 0
        let tr: MathSolver.Transport = { _, _, _ in
            calls += 1
            throw MathSolver.SolverError("HTTP_ERROR", "401")
        }
        XCTAssertThrowsError(try MathSolver(apiKey: "sk", transport: tr).solve("1+1")) { error in
            XCTAssertEqual((error as? MathSolver.SolverError)?.code, "HTTP_ERROR")
        }
        XCTAssertEqual(calls, 1)
    }

    func testStillWrongUnverified() throws {
        let tr: MathSolver.Transport = { _, _, _ in self.wrongCheck }
        let r = try MathSolver(apiKey: "sk", transport: tr).solve("2x+3=11")
        XCTAssertEqual(r.answer, 4, accuracy: 1e-9)
        XCTAssertFalse(r.verified)
        XCTAssertEqual(r.retries, 1)
    }

    // MARK: - HTTP-interface mock (fake HttpPost below the default transport)

    /// Serves scripted model contents / HTTP statuses; records every request.
    final class FakeHTTP {
        var n = 0
        var urls: [String] = []
        var auths: [String] = []
        var bodies: [String] = []
        let contents: [String]
        let statuses: [Int]
        init(contents: [String], statuses: [Int] = []) {
            self.contents = contents
            self.statuses = statuses
        }
    }

    /// JSON-escape a model reply so it can be embedded as the "content" string
    /// of a synthetic OpenAI-shaped response envelope.
    func envelope(_ content: String) -> String {
        let escaped = content
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        return #"{"choices":[{"message":{"content":"\#(escaped)"}}]}"#
    }

    func mockedSolver(_ apiKey: String, fake: FakeHTTP) throws -> MathSolver {
        try MathSolver(apiKey: apiKey, baseUrl: "https://mock.test/v1", model: "mock-model", httpPost: { url, headers, body in
            let i = fake.n
            fake.n += 1
            fake.urls.append(url)
            fake.auths.append(headers["Authorization"] ?? "")
            fake.bodies.append(body)
            let content = i < fake.contents.count ? fake.contents[i] : self.good
            let status = i < fake.statuses.count ? fake.statuses[i] : 200
            if status >= 300 { return (status, "upstream boom") }
            return (200, self.envelope(content))
        })
    }

    func testHTTPMockFullRoundTrip() throws {
        let fake = FakeHTTP(contents: [good])
        let solver = try mockedSolver("sk-mock", fake: fake)
        let r = try solver.solve("2x + 3 = 11, solve for x")
        XCTAssertTrue(r.verified)
        XCTAssertEqual(r.retries, 0)
        XCTAssertEqual(r.answer, 4, accuracy: 1e-9)
        XCTAssertEqual(fake.urls.count, 1)
        XCTAssertEqual(fake.urls[0], "https://mock.test/v1/chat/completions")
        XCTAssertEqual(fake.auths[0], "Bearer sk-mock")
        XCTAssertTrue(fake.bodies[0].contains(#""model":"mock-model""#))
        XCTAssertTrue(fake.bodies[0].contains(#""temperature":0"#))
        XCTAssertTrue(fake.bodies[0].contains(#""role":"system""#))
        XCTAssertTrue(fake.bodies[0].contains("STRICT JSON"))
    }

    func testHTTPMockCorrectiveRetry() throws {
        let fake = FakeHTTP(contents: [wrongCheck, good])
        let solver = try mockedSolver("sk", fake: fake)
        let r = try solver.solve("2x+3=11")
        XCTAssertTrue(r.verified)
        XCTAssertEqual(r.retries, 1)
        XCTAssertEqual(fake.urls.count, 2)
        XCTAssertTrue(fake.bodies[1].contains("failed verification"))
    }

    func testHTTPMockInvalidJSONReask() throws {
        let fake = FakeHTTP(contents: ["certainly not json", good])
        let solver = try mockedSolver("sk", fake: fake)
        let r = try solver.solve("1+1")
        XCTAssertTrue(r.verified)
        XCTAssertEqual(fake.urls.count, 2)
    }

    func testHTTPMock500() {
        let fake = FakeHTTP(contents: [], statuses: [500])
        XCTAssertThrowsError(try mockedSolver("sk", fake: fake).solve("1+1")) { error in
            XCTAssertEqual((error as? MathSolver.SolverError)?.code, "HTTP_ERROR")
        }
        XCTAssertEqual(fake.urls.count, 1)
    }

    func testHTTPMock401() {
        let fake = FakeHTTP(contents: [], statuses: [401])
        XCTAssertThrowsError(try mockedSolver("sk-bad", fake: fake).solve("1+1")) { error in
            XCTAssertEqual((error as? MathSolver.SolverError)?.code, "HTTP_ERROR")
        }
    }

    func testSmokeRealAPI() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipIf(env["SMOKE_API_KEY"] == nil, "smoke: set SMOKE_API_KEY to run")
        var base = env["SMOKE_BASE_URL"] ?? ""
        if base.isEmpty { base = "https://api.openai.com/v1" }
        var model = env["SMOKE_MODEL"]
        if model?.isEmpty ?? true { model = nil } // empty/unset -> gpt-4o-mini default
        let solver = try MathSolver(apiKey: env["SMOKE_API_KEY"]!, baseUrl: base, model: model)
        let r = try solver.solve("2x + 3 = 11, solve for x")
        print("smoke: answer=\(r.answer) verified=\(r.verified) retries=\(r.retries)")
        XCTAssertTrue(r.verified)
        XCTAssertEqual(r.answer, 4, accuracy: 1e-9)
    }
}
