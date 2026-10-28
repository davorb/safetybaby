import Foundation

struct Controls: Codable, Equatable {
    var armed: Bool
    var napMode: Bool
    enum CodingKeys: String, CodingKey { case armed, napMode = "nap_mode" }
}

struct Capabilities: Codable, Equatable {
    var vision: Bool
    var push: Bool
}

struct Reading: Codable, Equatable {
    var body: String
    var head: String
    var faceCovered: Bool
    var inCrib: Bool
    var eyes: String
    var confidence: Double
    enum CodingKeys: String, CodingKey {
        case body, head, eyes, confidence
        case faceCovered = "face_covered", inCrib = "in_crib"
    }
}

/// GET /api/state. Ages come from the server, never from the TV clock.
struct Situation: Codable, Equatable {
    var at: Date
    var controls: Controls
    var capabilities: Capabilities
    var cameraUp: Bool
    var cameraDownForS: Double
    var sleep: String
    var sleepForS: Double
    var observation: Reading?
    var observationAgeS: Double?
    var faceCoveredStreak: Int
    var stomachStreak: Int
    var headLeftForS: Double

    enum CodingKeys: String, CodingKey {
        case at, controls, capabilities, sleep, observation
        case cameraUp = "camera_up", cameraDownForS = "camera_down_for_s"
        case sleepForS = "sleep_for_s", observationAgeS = "observation_age_s"
        case faceCoveredStreak = "face_covered_streak", stomachStreak = "stomach_streak"
        case headLeftForS = "head_left_for_s"
    }
}

struct AlertInfo: Codable, Equatable {
    var kind: String
    var priority: String
    var at: Date
    var title: String
    var message: String
}

enum Tier: Int, Comparable {
    case normal, high, emergency
    static func < (a: Tier, b: Tier) -> Bool { a.rawValue < b.rawValue }

    init(_ priority: String) {
        self = switch priority {
        case "emergency": .emergency
        case "high": .high
        default: .normal
        }
    }
}

/// The union of event payloads, flattened: `type` says which fields are present.
struct EventBody: Codable, Equatable {
    var type: String
    var at: Date
    var previous: String?
    var current: String?
    var observation: Reading?
    var alert: AlertInfo?
    var delivered: Bool?
    var controls: Controls?
    var up: Bool?
    var detail: String?
}

struct ServerEvent: Codable, Equatable, Identifiable {
    var id: Int
    var event: EventBody
    var snapshotURL: String?
    var clipURL: String?
    enum CodingKeys: String, CodingKey {
        case id, event, snapshotURL = "snapshot_url", clipURL = "clip_url"
    }
}

struct TimelineSpan: Codable, Equatable {
    var state: String?
    var start: Date
    var end: Date
}

enum APIError: Error {
    case unreachable
    case unavailable
    case status(Int)
}

struct APIClient {
    var base: URL

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let plain = ISO8601DateFormatter()
        d.dateDecodingStrategy = .custom { decoder in
            let raw = try decoder.singleValueContainer().decode(String.self)
            if let date = fractional.date(from: raw) ?? plain.date(from: raw) { return date }
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: raw))
        }
        return d
    }()

    private static let session: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = 1.5
        c.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: c)
    }()

    private func request(_ path: String, query: [URLQueryItem] = [], method: String = "GET",
                         body: Data? = nil, timeout: TimeInterval = 1.5) async throws -> Data {
        var comps = URLComponents(url: base.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty { comps.queryItems = query }
        var req = URLRequest(url: comps.url!, timeoutInterval: timeout)
        req.httpMethod = method
        if let body {
            req.httpBody = body
            req.setValue("application/json", forHTTPHeaderField: "content-type")
        }
        let data: Data, response: URLResponse
        do { (data, response) = try await Self.session.data(for: req) } catch { throw APIError.unreachable }
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        if code == 503 { throw APIError.unavailable }
        guard (200..<300).contains(code) else { throw APIError.status(code) }
        return data
    }

    func state() async throws -> Situation {
        try Self.decoder.decode(Situation.self, from: await request("api/state"))
    }

    func frame() async throws -> Data { try await request("api/frame.jpg", timeout: 4) }

    func events(hours: Double, sinceID: Int? = nil) async throws -> [ServerEvent] {
        var q = [URLQueryItem(name: "hours", value: String(hours))]
        if let sinceID { q.append(URLQueryItem(name: "since_id", value: String(sinceID))) }
        return try Self.decoder.decode([ServerEvent].self, from: await request("api/events", query: q, timeout: 5))
    }

    func timeline(hours: Double = 24) async throws -> [TimelineSpan] {
        try Self.decoder.decode([TimelineSpan].self,
                                from: await request("api/timeline", query: [.init(name: "hours", value: String(hours))], timeout: 5))
    }

    func setControls(armed: Bool? = nil, napMode: Bool? = nil) async throws {
        var body: [String: Bool] = [:]
        if let armed { body["armed"] = armed }
        if let napMode { body["nap_mode"] = napMode }
        _ = try await request("api/controls", method: "POST", body: try JSONEncoder().encode(body))
    }

    func setReference() async throws {
        _ = try await request("api/reference", method: "POST", body: Data("{}".utf8), timeout: 5)
    }

    /// Server-relative URLs (snapshots, clips) become absolute. A `#t=` fragment survives.
    func absolute(_ path: String) -> URL? { URL(string: path, relativeTo: base)?.absoluteURL }
}
