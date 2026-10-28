import AVFoundation

/// Plays the bundled alert tones. The app cannot raise the TV volume, so Settings has a test button.
@MainActor
final class AlertSound {
    enum Tone: String {
        case emergency = "emergency-alarm"
        case high = "high-chime"
    }

    private var players: [Tone: AVAudioPlayer] = [:]

    init() {
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
    }

    func play(_ tone: Tone) {
        if players[tone] == nil, let url = Bundle.main.url(forResource: tone.rawValue, withExtension: "wav") {
            players[tone] = try? AVAudioPlayer(contentsOf: url)
            players[tone]?.volume = 1
        }
        guard let player = players[tone] else { return }
        player.currentTime = 0
        player.play()
    }

    func stop() { players.values.forEach { $0.stop() } }
}
