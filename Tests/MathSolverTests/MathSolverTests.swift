import XCTest
@testable import MathSolver

final class MathSolverTests: XCTestCase {
    let good = #"{"answer": 4, "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"], "verification": {"expression": "(11-3)/2"}}"#
    let wrong = #"{"answer": 4, "steps": ["..."], "verification": {"expression": "(11-3)/3"}}"#

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

    func testEvaluatorRejectsBadInput() {
        for bad in ["Process.launch()", "1+2)", "foo(1)", ""] {
            XCTAssertThrowsError(try MathSolver.evalExpression(bad))
        }
    }

    func testSolveVerifiedFirstTry() throws {
        var calls = 0
        var seenURL: String?, seenKey: String?
        let tr: MathSolver.Transport = { url, body, key in
            calls += 1; seenURL = url; seenKey = key
            return self.good
        }
        let r = try MathSolver.solve("2x + 3 = 11, solve for x", apiKey: "sk-test", transport: tr)
        XCTAssertTrue(r.verified)
        XCTAssertEqual(r.retries, 0)
        XCTAssertEqual(r.evaluated, 4)
        XCTAssertEqual(calls, 1)
        XCTAssertTrue(seenURL!.hasSuffix("/chat/completions"))
        XCTAssertEqual(seenKey, "sk-test")
    }

    func testSolveRetryRecovers() throws {
        var n = 0
        let tr: MathSolver.Transport = { _, _, _ in
            n += 1
            return n == 1 ? self.wrong : self.good
        }
        let r = try MathSolver.solve("2x+3=11", apiKey: "sk", transport: tr)
        XCTAssertTrue(r.verified)
        XCTAssertEqual(r.retries, 1)
    }

    func testInvalidJSONThenOK() throws {
        var n = 0
        let tr: MathSolver.Transport = { _, _, _ in
            n += 1
            return n == 1 ? "no json" : self.good
        }
        let r = try MathSolver.solve("1+1", apiKey: "sk", transport: tr)
        XCTAssertTrue(r.verified)
    }

    func testInvalidTwiceRaises() {
        XCTAssertThrowsError(try MathSolver.solve("1+1", apiKey: "sk", transport: { _, _, _ in "nothing" })) { error in
            XCTAssertEqual((error as? MathSolver.SolverError)?.code, "INVALID_JSON")
        }
    }

    func testNoAPIKey() {
        XCTAssertThrowsError(try MathSolver.solve("1+1", apiKey: "")) { error in
            XCTAssertEqual((error as? MathSolver.SolverError)?.code, "NO_API_KEY")
        }
    }

    func testHTTPErrorNoRetry() {
        var calls = 0
        let tr: MathSolver.Transport = { _, _, _ in
            calls += 1
            throw MathSolver.SolverError("HTTP_ERROR", "401")
        }
        XCTAssertThrowsError(try MathSolver.solve("1+1", apiKey: "sk", transport: tr)) { error in
            XCTAssertEqual((error as? MathSolver.SolverError)?.code, "HTTP_ERROR")
        }
        XCTAssertEqual(calls, 1)
    }

    func testStillWrongUnverified() throws {
        let tr: MathSolver.Transport = { _, _, _ in self.wrong }
        let r = try MathSolver.solve("2x+3=11", apiKey: "sk", transport: tr)
        XCTAssertFalse(r.verified)
        XCTAssertEqual(r.retries, 1)
    }
}
