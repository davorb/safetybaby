import SwiftUI

extension Color {
    init(hex: UInt32) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255
        )
    }
}

/// Night-first palette from the design's visual-language board.
enum Theme {
    static let ground = Color(hex: 0x0D0C0B)
    static let panel = Color(hex: 0x16140F)
    static let raised = Color(hex: 0x1D1A15)
    static let text = Color(hex: 0xE4DDD0)
    static let muted = Color(hex: 0xADA493)
    static let focus = Color(hex: 0xF2E6CC)

    static let asleep = Color(hex: 0x8C9CF5)
    static let stirring = Color(hex: 0xBFA3F0)
    static let awake = Color(hex: 0xE2C77E)
    static let crying = Color(hex: 0xF69A73)

    static let emergencyGround = Color(hex: 0x5C0A0D)
    static let emergencyEdge = Color(hex: 0xFF6B5E)
    static let emergencyText = Color(hex: 0xFFEEE9)
    static let emergencyMuted = Color(hex: 0xFFC8BD)
    static let high = Color(hex: 0xFFB43D)
    static let highText = Color(hex: 0x241800)
    static let normal = Color(hex: 0x1B2130)
    static let normalText = Color(hex: 0xC2D0EF)
    static let ok = Color(hex: 0x8ED6A5)
    static let nap = Color(hex: 0x8FB4FF)

    static func sleep(_ state: String) -> Color {
        switch state {
        case "asleep": asleep
        case "stirring": stirring
        case "awake": awake
        case "crying": crying
        default: muted
        }
    }
}

extension Font {
    static func ui(_ size: CGFloat, bold: Bool = false) -> Font {
        .system(size: size, weight: bold ? .bold : .regular).monospacedDigit()
    }
}

/// Focused item: scale 1.06, a cream ring with a halo, deeper shadow, 150 ms ease-out.
struct FocusRing: ViewModifier {
    let focused: Bool
    var radius: CGFloat = 32

    func body(content: Content) -> some View {
        content
            .scaleEffect(focused ? 1.06 : 1)
            .overlay(
                RoundedRectangle(cornerRadius: radius)
                    .strokeBorder(Theme.focus, lineWidth: focused ? 5 : 0)
                    .padding(-2)
            )
            .background(
                RoundedRectangle(cornerRadius: radius + 6)
                    .fill(Theme.focus.opacity(focused ? 0.3 : 0))
                    .padding(-12)
            )
            .shadow(color: .black.opacity(focused ? 0.65 : 0), radius: 30, y: 24)
            .animation(.easeOut(duration: 0.15), value: focused)
    }
}

/// Button style that draws the design's focus treatment instead of the system platter.
struct FocusStyle: ButtonStyle {
    var radius: CGFloat = 32

    func makeBody(configuration: Configuration) -> some View {
        Inner(configuration: configuration, radius: radius)
    }

    private struct Inner: View {
        let configuration: ButtonStyle.Configuration
        let radius: CGFloat
        @Environment(\.isFocused) private var focused

        var body: some View {
            configuration.label.modifier(FocusRing(focused: focused, radius: radius))
        }
    }
}

enum Format {
    static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        if s < 60 { return "\(s) s" }
        if s < 3600 { return s % 60 == 0 || s >= 600 ? "\(s / 60) min" : "\(s / 60) min \(s % 60) s" }
        let m = (s % 3600) / 60
        return m == 0 ? "\(s / 3600) h" : "\(s / 3600) h \(m) min"
    }

    static func ago(_ seconds: Double) -> String { "\(duration(seconds)) ago" }

    static func clock(_ date: Date) -> String {
        date.formatted(.dateTime.hour(.twoDigits(amPM: .omitted)).minute(.twoDigits))
    }

    static func mmss(_ seconds: Double) -> String {
        let s = max(0, Int(seconds.rounded()))
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
