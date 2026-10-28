import SwiftUI

/// The camera still, dimmed and frozen when the picture is stale.
struct CameraView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Theme.ground
            .overlay {
                if let frame = model.frame {
                    Image(uiImage: frame)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .saturation(stale ? 0 : 1)
                        .opacity(stale ? 0.45 : 1)
                } else {
                    Text("Waiting for the first camera frame")
                        .font(.ui(26)).foregroundStyle(Theme.muted)
                }
            }
            .clipped()
    }

    private var stale: Bool { model.frameFailed || model.connection == .unreachable }
}

struct Chip: View {
    let label: String
    var color: Color = Theme.ok
    var symbol: String?
    var outlined = false
    var off = false

    var body: some View {
        HStack(spacing: 14) {
            if let symbol {
                Image(systemName: symbol).font(.system(size: 30, weight: .bold)).foregroundStyle(color)
            } else {
                Circle().fill(off ? Theme.high : color).frame(width: 16, height: 16)
            }
            Text(label).font(.ui(34, bold: true)).foregroundStyle(Theme.text)
            if off { Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Theme.high).font(.system(size: 26)) }
        }
        .padding(.horizontal, 28).frame(height: 68)
        .background(Capsule().fill(Theme.ground.opacity(0.72)))
        .overlay(Capsule().strokeBorder(color, lineWidth: outlined ? 3 : 0))
    }
}

struct ChipRow: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        HStack(spacing: 20) {
            if let s = model.state {
                if s.controls.armed {
                    Chip(label: "Armed", color: Theme.ok, symbol: "checkmark.shield", outlined: true)
                } else {
                    Chip(label: "Disarmed", color: Theme.high, symbol: "shield.slash", outlined: true)
                }
                Chip(label: s.cameraUp ? "Camera" : "Camera down", off: !s.cameraUp)
                Chip(label: s.capabilities.vision ? "Vision" : "Vision off", off: !s.capabilities.vision)
                Chip(label: s.capabilities.push ? "Push" : "Push off", off: !s.capabilities.push)
            }
        }
    }
}

struct SleepBlock: View {
    let state: String
    let seconds: Double
    var compact = false

    var body: some View {
        let color = Theme.sleep(state)
        VStack(alignment: .leading, spacing: compact ? 8 : 14) {
            HStack(spacing: compact ? 22 : 28) {
                Circle().fill(color).frame(width: compact ? 30 : 40, height: compact ? 30 : 40)
                Text(state.capitalized)
                    .font(.ui(compact ? 84 : 132, bold: true)).foregroundStyle(color)
                    .overlay(alignment: .bottom) {
                        if state == "crying" { Hatch(color: color).frame(height: 8).clipShape(Capsule()) }
                    }
            }
            Text("for \(Format.duration(seconds))")
                .font(.ui(compact ? 40 : 50)).foregroundStyle(Theme.text)
                .padding(.leading, compact ? 52 : 68)
        }
    }
}

/// Diagonal hatching: crying and disarmed never rely on color alone.
struct Hatch: View {
    var color: Color
    var body: some View {
        Canvas { ctx, size in
            var x: CGFloat = -size.height
            while x < size.width {
                var p = Path()
                p.move(to: CGPoint(x: x, y: size.height))
                p.addLine(to: CGPoint(x: x + size.height, y: 0))
                ctx.stroke(p, with: .color(color), lineWidth: 4)
                x += 14
            }
        }
    }
}

struct ReadingValue: View {
    let label: String
    let value: String
    var uncertain = false
    var dim = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.ui(28)).foregroundStyle(Theme.muted)
            Text(uncertain ? value + "?" : value)
                .font(.ui(40, bold: true)).italic(uncertain)
                .foregroundStyle(Theme.text.opacity(dim ? 0.5 : 1))
        }
    }
}

struct PositionCard: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            if model.state?.capabilities.vision == false {
                visionOff
            } else if let o = model.state?.observation, let age = model.state?.observationAgeS {
                reading(o, age: age)
            } else {
                HStack {
                    Text("Latest position check").font(.ui(30, bold: true))
                    Spacer()
                    Text("No check yet").font(.ui(28)).foregroundStyle(Theme.muted)
                }
            }
        }
        .foregroundStyle(Theme.text)
        .padding(.horizontal, 36).padding(.top, 30).padding(.bottom, 34)
        .frame(width: 780, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 32).fill(Theme.ground.opacity(0.74)))
    }

    private var visionOff: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Position checks are off", systemImage: "eye.slash").font(.ui(34, bold: true))
            Text("Face covered, stomach and head-left alerts cannot fire. Add the vision key on the server.")
                .font(.ui(28)).foregroundStyle(Theme.muted)
        }
    }

    private func reading(_ o: Reading, age: Double) -> some View {
        let late = age > 16 * 60
        let low = o.confidence < 0.5
        let percent = Int((o.confidence * 100).rounded())
        return VStack(alignment: .leading, spacing: 24) {
            HStack(alignment: .firstTextBaseline) {
                Text("Latest position check").font(.ui(30, bold: true))
                Spacer()
                HStack(spacing: 8) {
                    Text(Format.ago(age)).foregroundStyle(late ? Theme.high : Theme.muted)
                    if late { Text("late").foregroundStyle(Theme.high).bold() }
                    Text("· \(percent) % sure").foregroundStyle(low ? Theme.high : Theme.muted)
                }.font(.ui(28))
            }
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 24, alignment: .leading), count: 3), alignment: .leading, spacing: 26) {
                ReadingValue(label: "Body", value: Self.body(o.body), uncertain: low, dim: late)
                ReadingValue(label: "Head", value: Self.head(o.head), uncertain: low, dim: late)
                ReadingValue(label: "Face", value: o.faceCovered ? "Covered" : "Clear", uncertain: low, dim: late)
                ReadingValue(label: "Where", value: Self.location(o.location), dim: late)
                ReadingValue(label: "Eyes", value: o.eyes.capitalized, dim: late)
            }
            if low {
                Text("Below 50 %. This reading cannot trigger an alert.").font(.ui(26)).foregroundStyle(Theme.high)
            }
        }
    }

    static func location(_ raw: String) -> String {
        switch raw {
        case "crib": "In crib"
        case "parents_bed": "In our bed"
        case "held": "Held"
        case "not_visible": "Not visible"
        default: raw.capitalized
        }
    }

    static func body(_ raw: String) -> String {
        switch raw {
        case "back": "On back"
        case "stomach": "Stomach"
        case "side_left": "Left side"
        case "side_right": "Right side"
        default: "Unclear"
        }
    }

    static func head(_ raw: String) -> String { raw == "unknown" ? "Unclear" : raw.capitalized }
}

/// Amber outlined card for a danger that has started but not yet become an alert.
struct PendingCard: View {
    let kicker: String
    let title: String
    let progress: Double
    let left: String
    let right: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(kicker).font(.ui(26, bold: true)).foregroundStyle(Theme.high).tracking(1.5)
            Text(title).font(.ui(46, bold: true)).foregroundStyle(Theme.text)
            ProgressView(value: min(max(progress, 0), 1))
                .tint(Theme.high)
                .scaleEffect(x: 1, y: 2.4)
                .padding(.vertical, 6)
            HStack {
                Text(left).font(.ui(30)).foregroundStyle(Theme.text)
                Spacer()
                Text(right).font(.ui(30)).foregroundStyle(Theme.muted)
            }
        }
        .padding(.horizontal, 32).padding(.vertical, 26)
        .frame(width: 720)
        .background(RoundedRectangle(cornerRadius: 28).fill(Theme.ground.opacity(0.82)))
        .overlay(RoundedRectangle(cornerRadius: 28).strokeBorder(Theme.high, lineWidth: 4))
    }
}

struct PendingCards: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let s = model.state {
            VStack(alignment: .leading, spacing: 20) {
                if s.faceCoveredStreak > 0 || s.stomachStreak > 0 {
                    let streak = max(s.faceCoveredStreak, s.stomachStreak)
                    PendingCard(
                        kicker: "WATCHING · EMERGENCY IF CONFIRMED",
                        title: s.faceCoveredStreak >= s.stomachStreak ? "Possible face covered" : "Possible face-down",
                        progress: Double(streak) / 2,
                        left: "Check \(streak) of 2", right: "Rechecking within 15 s"
                    )
                }
                if s.headLeftForS > 0 {
                    PendingCard(
                        kicker: "WATCHING · ALERT AT 2:00",
                        title: "Head turned left",
                        progress: s.headLeftForS / 120,
                        left: "\(Format.mmss(s.headLeftForS)) of 2:00",
                        right: "\(Format.mmss(120 - s.headLeftForS)) left"
                    )
                }
            }
        }
    }
}

struct NoticeStrip: View {
    let text: String
    var body: some View {
        HStack(spacing: 18) {
            Image(systemName: "info.circle").font(.system(size: 34))
            Text(text).font(.ui(32, bold: true))
        }
        .foregroundStyle(Theme.normalText)
        .padding(.horizontal, 32).frame(height: 76)
        .background(Capsule().fill(Theme.normal))
        .transition(.opacity)
    }
}
