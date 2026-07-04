# *සිංහල* &mdash; Sinhala (`si`)

This datasheet is for sps-corpus-4.0-2026-06-12 of the Mozilla Common Voice *Spontaneous Speech* dataset for Sinhala [සිංහල - `si`]. The dataset contains 59 clips representing 0.4 hours of recorded speech (0 hours validated) from 1 speakers.

## Data splits for modelling

The dataset clips are categorised by transcription status and training-set assignment. The following tables summarise the distribution.

### Audio clips

| Bucket | Clips | % |
| --- | --- | --- |
| Transcribed & Validated | 0 | 0.0% |
| Transcribed & Pending | 0 | 0.0% |
| Not transcribed | 59 | 100.0% |

### Training splits

| Bucket | Clips | % |
| --- | --- | --- |
| Train | 0 | 0.0% |
| Dev | 0 | 0.0% |
| Test | 0 | 0.0% |
| Unassigned | 59 | 100.0% |

Training split coverage: 0 of 0 transcribed & validated clips (0.0%)

## Transcriptions

### Transcription status

| Bucket | Clips | % |
| --- | --- | --- |
| Validated | 0 | 0.0% |
| Pending | 0 | 0.0% |
| Edited | 0 | 0.0% |

### Samples

#### Questions

There follows a randomly selected sample of questions used in the corpus.

1. *ඔබ දන්නා අය අතර වැඩි ඉල්ලුමක් ඇති වෘත්තීය අධ්‍යාපන පාඨමාලා මොනවාද?*
2. *එළිමහන් සංගීත සංදර්ශන වලට යාමට ඔබ කැමති හෝ අකමැති හේතු කවරේද?*
3. *ඔබගේ නිවසේ සහ ඔබ වාසය කරන ප්‍රදේශයේ මැසි මදුරුවන් මර්දනය කිරීමට යොදාගන්නා ක්‍රියාමාර්ග මොනවාද?*
4. *ප්ලාස්ටික්, ලෝහ වැනි දෑ ප්‍රතිචක්‍රීකරණය කිරීම සඳහා ඔබ පදිංචි ප්‍රදේශයේ අනුගමනය කරන ක්‍රියාමාර්ග ප්‍රමාණවත් යැයි ඔබ සිතන්නේද? ඒ ඇයි?*
5. *කාසි හා නෝට්ටු හැරුණුවිට මුදල් ගෙවීම හෝ ලබාගැනීම සඳහා ඔබ යොදාගන්නා ක්‍රම මොනවාද?*

#### Responses

There follows a randomly selected sample of transcribed responses from the corpus.

### Fields

Each row of a `tsv` file represents a single audio clip, and contains the following information:

- `client_id` - hashed UUID of a given user
- `audio_id` - numeric id for audio file
- `audio_file` - audio file name
- `duration_ms` - duration of audio in milliseconds
- `prompt_id` - numeric id for prompt
- `prompt` - question for user
- `transcription` - transcription of the audio response
- `votes` - number of people that who approved a given transcript
- `age` - age of the speaker[^1]
- `gender` - gender of the speaker[^1]
- `language` - language name
- `split` - for data modelling, which subset of the data does this clip pertain to
- `char_per_sec` - how many characters of transcription per second of audio
- `quality_tags` - some automated assessment of the transcription--audio pair, separated by `|`
  - `transcription-length` - character per second under 3 characters per second
  - `speech-rate` - characters per second over 30 characters per second
  - `short-audio` - audio length under 2 seconds
  - `long-audio` - audio length over 5 minutes
  - `non-allowed-script` - transcription contains characters from a writing system not associated with the language
  - `mixed-script-words` - a single word contains characters from multiple writing systems
  - `mixed-script-transcription` - transcription spans multiple writing systems, but each word consistently uses only one

---

[^1]: For a full list of age, gender, and accent options, see the [demographics spec](https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts). These will only be reported if the speaker opted in to provide that information.

## Get involved

### Community links

- [Common Voice translators on Pontoon](https://pontoon.mozilla.org/si/common-voice/contributors/)
- [Common Voice Communities](https://github.com/common-voice/common-voice/blob/main/docs/COMMUNITIES.md)

### Discussions

- [Common Voice on Matrix](https://chat.mozilla.org/#/room/#common-voice:mozilla.org)
- [Common Voice on Discourse](https://discourse.mozilla.org/t/about-common-voice-readme-first/17218)
- [Common Voice on Discord](https://discord.gg/9QTj9zwn)
- [Common Voice on Telegram](https://t.me/mozilla_common_voice)

### Contribute

- [Contribute questions](https://commonvoice.mozilla.org/spontaneous-speech/beta/question)
- [Validate questions](https://commonvoice.mozilla.org/spontaneous-speech/beta/validate)
- [Answer questions](https://commonvoice.mozilla.org/spontaneous-speech/beta/prompts)
- [Transcribe recordings](https://commonvoice.mozilla.org/spontaneous-speech/beta/transcribe)
- [Validate transcriptions](https://commonvoice.mozilla.org/spontaneous-speech/beta/check-transcript)

## Licence

This dataset is released under the [Creative Commons Zero (CC-0)](https://creativecommons.org/public-domain/cc0/) licence. By downloading this data you agree to not determine the identity of speakers in the dataset.
