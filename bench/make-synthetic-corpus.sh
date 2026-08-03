#!/usr/bin/env bash
# Build a bench corpus aimed at ONE question: does a hotword list make Voxtral spell a name that is
# also a French common word?
#
# Ground truth is exact because we author it — `say` speaks the reference verbatim. The cost of that
# is clean studio speech, so the absolute WER here does NOT transfer to noisy phone Opus. It does not
# need to: the failure under test is a *spelling* choice made from phonetics the TTS reproduces
# faithfully (`Loup` and `Lou` are homophones for a French speaker and for the synthesiser alike).
#
# Every sentence mixes both classes so one run scores both:
#   class A — the name IS a common word:      Loup, Colombe, Rose, Olivier, Pierre
#   class B — control, not a common word:     Grenoble, Chamonix, Mathilde, Thibault, Fabien, Annecy
set -euo pipefail

OUT="${1:?usage: make-corpus.sh <dir>}"
mkdir -p "$OUT"

VOICES=(Thomas Jacques "Flo (French (France))" "Shelley (French (France))" "Rocko (French (France))")

write() {  # write <n> <bias-csv> <text>
  local n="$1" bias="$2" text="$3"
  local voice="${VOICES[$(( (10#$n - 1) % ${#VOICES[@]} ))]}"
  printf '%s' "$text" > "$OUT/note$n.txt"
  printf '%s\n' "${bias//,/$'\n'}" > "$OUT/note$n.bias.txt"
  say -v "$voice" -o "$OUT/note$n.aiff" "$text"
  # WhatsApp's own shape: mono, 16 kHz, ~16 kbps Opus in Ogg. Transcoding to anything richer would
  # measure a recording this endpoint will never actually receive.
  ffmpeg -loglevel error -y -i "$OUT/note$n.aiff" -ac 1 -ar 16000 -c:a libopus -b:a 16k "$OUT/note$n.ogg"
  rm -f "$OUT/note$n.aiff"
}

write 01 "Loup,Mathilde,Grenoble" \
  "Salut Loup, c'est Mathilde. On se retrouve à Grenoble demain matin vers neuf heures."
write 02 "Colombe,Loup,Chamonix" \
  "Coucou, ici Colombe. Est-ce que Loup a réservé le refuge à Chamonix pour le week-end ?"
write 03 "Olivier,Thibault,Loup" \
  "Bonjour Olivier, Thibault m'a dit que Loup rentrait mardi. Tu confirmes ?"
write 04 "Rose,Pierre,Loup" \
  "Rose a laissé les clés chez Pierre. Loup passera les chercher ce soir."
write 05 "Fabien,Loup,Colombe" \
  "Fabien, dis à Loup que le livre de Colombe est arrivé à la librairie."
write 06 "Loup,Mathilde,Annecy,Rose" \
  "Loup et Mathilde partent à Annecy samedi. Rose les rejoint dimanche."
write 07 "Pierre,Olivier,Loup" \
  "J'ai croisé Pierre et Olivier au marché. Ils demandaient des nouvelles de Loup."
write 08 "Thibault,Chamonix,Loup" \
  "Thibault m'appelle de Chamonix. Il cherche Loup depuis ce matin."
write 09 "Colombe,Loup,Rose" \
  "Colombe voudrait savoir si Loup vient au dîner chez Rose vendredi."
write 10 "Loup,Grenoble,Fabien" \
  "Bonjour, c'est Loup. Je suis à Grenoble, je rappelle Fabien dans dix minutes."

echo "corpus: $(ls "$OUT"/*.ogg | wc -l | tr -d ' ') recordings"
for f in "$OUT"/*.ogg; do
  printf '%s  %ss  %sB\n' "$(basename "$f")" \
    "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" | cut -c1-4)" \
    "$(stat -f%z "$f")"
done
