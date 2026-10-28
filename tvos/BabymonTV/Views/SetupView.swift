import SwiftUI
import Darwin

struct SetupView: View {
    @Environment(AppModel.self) private var model
    @State private var address = "http://"
    @State private var found: [String] = []
    @State private var scanning = false
    @State private var checks: [Check] = []
    @FocusState private var field: Bool

    struct Check: Identifiable { let id = UUID(); let text: String; let sub: String?; let ok: Bool }

    var body: some View {
        ZStack {
            Theme.ground.ignoresSafeArea()
            HStack(alignment: .top, spacing: 80) {
                VStack(alignment: .leading, spacing: 0) {
                    Text("Connect to babymon").font(.ui(92, bold: true))
                    Text("Enter the address of the babymon server on your home network.")
                        .font(.ui(40)).foregroundStyle(Theme.muted).padding(.top, 26)
                    Text("Server address").font(.ui(30, bold: true)).padding(.top, 52)
                    TextField("http://192.168.1.24:8080", text: $address)
                        .font(.ui(48, bold: true)).keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                        .focused($field).onSubmit { Task { await test() } }
                        .padding(.top, 14)
                    Text("Select to open the keyboard. Port 8080 is the default.").font(.ui(28)).foregroundStyle(Theme.muted).padding(.top, 14)

                    Text("Found on your network").font(.ui(30, bold: true)).padding(.top, 44)
                    VStack(spacing: 12) {
                        ForEach(found, id: \.self) { host in
                            Button { address = "http://\(host)"; Task { await test() } } label: {
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text("babymon").font(.ui(38, bold: true))
                                        Text(host).font(.ui(28)).foregroundStyle(Theme.muted)
                                    }
                                    Spacer()
                                    Text("Use this").font(.ui(32, bold: true)).foregroundStyle(Theme.ok)
                                }
                                .padding(.horizontal, 36).frame(height: 104)
                                .background(RoundedRectangle(cornerRadius: 26).fill(Theme.raised))
                            }.buttonStyle(FocusStyle(radius: 26))
                        }
                        if scanning {
                            Label("Still looking", systemImage: "ellipsis").font(.ui(30)).foregroundStyle(Theme.muted)
                        } else if found.isEmpty {
                            Button { Task { await scan() } } label: {
                                Text("Scan for port 8080").font(.ui(32, bold: true)).padding(.horizontal, 36).frame(height: 84)
                                    .background(Capsule().fill(Theme.raised))
                            }.buttonStyle(FocusStyle(radius: 42))
                        }
                    }.padding(.top, 14)
                }.frame(maxWidth: .infinity, alignment: .leading)

                VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Checking the connection").font(.ui(42, bold: true)).padding(.bottom, 10)
                        if checks.isEmpty { Text("Not tested yet.").font(.ui(32)).foregroundStyle(Theme.muted) }
                        ForEach(checks) { c in
                            HStack(spacing: 24) {
                                Image(systemName: c.ok ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                                    .font(.system(size: 44)).foregroundStyle(c.ok ? Theme.ok : Theme.high)
                                VStack(alignment: .leading) {
                                    Text(c.text).font(.ui(38))
                                    if let s = c.sub { Text(s).font(.ui(28)).foregroundStyle(Theme.muted) }
                                }
                            }.padding(.vertical, 14)
                        }
                    }
                    .padding(.horizontal, 44).padding(.vertical, 36).frame(width: 640, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 36).fill(Color(hex: 0x17140F)))
                    Label("This link is not encrypted and has no password. Use it on your home network only.", systemImage: "lock.open")
                        .font(.ui(30)).padding(28).frame(width: 640, alignment: .leading)
                        .overlay(RoundedRectangle(cornerRadius: 28).strokeBorder(Color(hex: 0x3A352C), lineWidth: 3))
                    Spacer()
                    Button { Task { await test() } } label: {
                        Text(checks.isEmpty ? "Test connection" : "Start monitoring").font(.ui(44, bold: true))
                            .foregroundStyle(Color(hex: 0x17140F)).padding(.horizontal, 64).frame(height: 112)
                            .background(Capsule().fill(Theme.text))
                    }.buttonStyle(FocusStyle(radius: 56))
                }
            }
            .foregroundStyle(Theme.text)
            .padding(.horizontal, 30).padding(.top, 20)
        }
        .task { await scan() }
    }

    private func normalized() -> URL? {
        var s = address.trimmingCharacters(in: .whitespaces)
        if !s.hasPrefix("http") { s = "http://" + s }
        guard var c = URLComponents(string: s), c.host != nil else { return nil }
        if c.port == nil { c.port = 8080 }
        c.path = ""
        return c.url
    }

    private func test() async {
        guard let url = normalized() else {
            checks = [Check(text: "That address does not look right", sub: "Example: http://192.168.1.24:8080", ok: false)]
            return
        }
        let api = APIClient(base: url)
        var out: [Check] = []
        do {
            let s = try await api.state()
            out.append(Check(text: "Server reached", sub: nil, ok: true))
            out.append(Check(text: "Camera picture", sub: s.cameraUp ? nil : "Camera is down right now", ok: s.cameraUp))
            out.append(Check(text: s.capabilities.vision ? "Position checks on" : "Position checks off",
                             sub: s.capabilities.vision ? nil : "No Anthropic key on the server", ok: s.capabilities.vision))
            out.append(Check(text: s.capabilities.push ? "Phone alerts on" : "Phone alerts off",
                             sub: s.capabilities.push ? nil : "No push key on the server", ok: s.capabilities.push))
            checks = out
            model.connect(to: url)
        } catch APIError.unavailable {
            checks = [Check(text: "Server reached", sub: "It is still starting. Try again in a moment.", ok: false)]
        } catch {
            checks = [Check(text: "Could not reach the server", sub: "Check the address and that babymon is running.", ok: false)]
        }
    }

    private func scan() async {
        guard let prefix = Self.localPrefix() else { return }
        scanning = true
        found = []
        await withTaskGroup(of: String?.self) { group in
            var next = 1
            var running = 0
            while next < 255 || running > 0 {
                if next < 255, running < 48 {
                    let host = "\(prefix).\(next)"
                    next += 1; running += 1
                    group.addTask { await Self.probe(host) }
                } else if let result = await group.next() {
                    running -= 1
                    if let result { found.append(result) }
                }
            }
        }
        scanning = false
    }

    private static func probe(_ host: String) async -> String? {
        var req = URLRequest(url: URL(string: "http://\(host):8080/api/state")!, timeoutInterval: 0.7)
        req.cachePolicy = .reloadIgnoringLocalCacheData
        guard let (_, resp) = try? await URLSession.shared.data(for: req),
              let code = (resp as? HTTPURLResponse)?.statusCode, code == 200 || code == 503 else { return nil }
        return "\(host):8080"
    }

    /// First three octets of this device's LAN address.
    private static func localPrefix() -> String? {
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return nil }
        defer { freeifaddrs(ifaddr) }
        for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let i = ptr.pointee
            guard i.ifa_addr.pointee.sa_family == UInt8(AF_INET), String(cString: i.ifa_name).hasPrefix("en") else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            getnameinfo(i.ifa_addr, socklen_t(i.ifa_addr.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST)
            let parts = String(cString: host).split(separator: ".")
            if parts.count == 4, parts[0] != "169" { return parts.prefix(3).joined(separator: ".") }
        }
        return nil
    }
}
