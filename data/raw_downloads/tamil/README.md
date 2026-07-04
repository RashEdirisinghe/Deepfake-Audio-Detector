# *தமிழ்* &mdash; Tamil (`ta`)

This datasheet is for cv-corpus-26.0-2026-06-12 of the Mozilla Common Voice *Scripted Speech* dataset for Tamil [தமிழ் - `ta`]. The dataset contains 247671 clips representing 425.33 hours of recorded speech (234.79 hours validated) from 981 speakers, recorded from a text corpus of 118,842 sentences.

## Language

### Accents

| Code | Accent | Clips | Speakers |
|---|---|---|---|
| - |  | 5,139 (2.1%) | 50 (5.1%) |

## Demographic information

The dataset includes the following self-declared age and gender distributions. A coverage summary is shown below each table.

### Gender

Self-declared gender information. The table shows clip and speaker counts with percentages. Speakers who did not declare a gender are listed as Unspecified. A dash (-) indicates zero.

| Code | Gender | Clips | Speakers |
|---|---|---|---|
| male_masculine | Male, masculine | 46,918 (18.9%) | 306 (31.2%) |
| female_feminine | Female, feminine | 38,203 (15.4%) | 137 (14.0%) |
| transgender | Transgender | - | - |
| non-binary | Non-binary | - | - |
| do_not_wish_to_say | Prefer not to say | - | - |
| - | Unspecified | 162,550 (65.6%) | 642 (65.4%) |

*Gender declared: 85,121 of 247,671 clips (34.4%), 339 of 981 speakers (34.6%)*

### Age

Self-declared age information. The table shows clip and speaker counts with percentages. Speakers who did not declare an age are listed as Unspecified. A dash (-) indicates zero.

| Code | Age | Clips | Speakers |
|---|---|---|---|
| teens | Teens | 9,429 (3.8%) | 24 (2.4%) |
| twenties | Twenties | 26,825 (10.8%) | 240 (24.5%) |
| thirties | Thirties | 29,463 (11.9%) | 112 (11.4%) |
| fourties | Fourties | 6,688 (2.7%) | 52 (5.3%) |
| fifties | Fifties | 6,090 (2.5%) | 24 (2.4%) |
| sixties | Sixties | 748 (0.3%) | 4 (0.4%) |
| seventies | Seventies | 4,814 (1.9%) | 4 (0.4%) |
| eighties | Eighties | 85 (0.0%) | 1 (0.1%) |
| nineties | Nineties | - | - |
| - | Unspecified | 163,529 (66.0%) | 627 (63.9%) |

*Age declared: 84,142 of 247,671 clips (34.0%), 354 of 981 speakers (36.1%)*

## Data splits for modelling

**Clip buckets**

| Bucket | Clips |
|---|---|
| Validated | 136,719 (55.2%) |
| Invalidated | 5,749 (2.3%) |
| Other | 105,203 (42.5%) |

**Training splits**

| Split | Clips |
|---|---|
| Train | 46,519 (34.0%) |
| Dev | 12,168 (8.9%) |
| Test | 12,241 (9.0%) |

*Training split coverage: 70,928 of 136,719 validated clips (51.9%)*

The dataset contains 136719 validated, 5749 invalidated, and 105203 unresolved clips. The average clip duration is 6.182 seconds.

## Text corpus

**Validated sentences:** 118,292

| Category | Count |
|---|---|
| Unvalidated sentences | 550 |
| Pending sentences | 549 |
| Rejected sentences | 1 |
| Reported sentences | 3,437 |

The corpus contains 118,842 sentences: 118,292 validated and 550 unvalidated (549 pending review, 1 rejected), with 3,437 reported for review.

### Sample

There follows a randomly selected sample of five sentences from the corpus.

1. *அமைவாய் விளக்கை அங்கையில் தூக்கிச்*
2. *பல காலம் போராடித் தேவர்களுக்கு அவர்களுடைய ராஜ்யத்தைத் திரும்பவும் வாங்கிக் கொடுத்த கருணையாளன் அல்லவா நீ?*
3. *அவனுக்கு இன்.அமுது படைத்து அருகிருந்து பரிமாறும் நிலையில் அவள் அன்பு வெளிப் படுகிறது.*
4. *அவனைக் கண்டு ஒவ்வோர் அரசியல் தலைவனும் வெட்கப்படல் வேண்டும்.*
5. *அதன் காரணமாகக் கெட்ட போரிடும் உலகமே தோன்றியுள்ளது.*

### Sources

| Source | Sentences |
|---|---|
| wikisource-ta | 99,615 (84.2%) |
| covost2-en_ta | 12,170 (10.3%) |
| sentence-collector | 6,115 (5.2%) |
| Other | 392 (0.3%) |

### Fields

#### Clips

Each row of a `tsv` file represents a single audio clip, and contains the following information:

- `client_id` - hashed UUID of a given user
- `path` - relative path of the audio file
- `sentence` - the sentence to be read aloud
- `sentence_id` - unique identifier for the sentence
- `sentence_domain` - domain classification(s) of the sentence
- `up_votes` - number of people who said audio matches the text
- `down_votes` - number of people who said audio does not match text
- `age` - age of the speaker[^1]
- `gender` - gender of the speaker[^1]
- `accents` - accents of the speaker[^1]
- `variant` - variant of the language[^1]
- `locale` - locale code of the language
- `segment` - if sentence belongs to a custom dataset segment, it will be listed here

[^1]: For a full list of age, gender, and accent options, see the [demographics spec](https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts). These will only be reported if the speaker opted in to provide that information.

#### `validated_sentences.tsv`

The `validated_sentences.tsv` file contains one row per validated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `is_used` - whether the sentence is still in circulation for recording
- `clips_count` - number of clips recorded for this sentence

#### `unvalidated_sentences.tsv`

The `unvalidated_sentences.tsv` file contains one row per unvalidated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `up_votes` - number of upvotes the sentence received
- `down_votes` - number of downvotes the sentence received
- `status` - current status of the sentence (`pending` or `rejected`)

## Get involved

### Community links

- [Common Voice translators on Pontoon](https://pontoon.mozilla.org/ta/common-voice/contributors/)
- [Common Voice Communities](https://github.com/common-voice/common-voice/blob/main/docs/COMMUNITIES.md)

### Discussions

- [Common Voice on Matrix](https://chat.mozilla.org/#/room/#common-voice:mozilla.org)
- [Common Voice on Discourse](https://discourse.mozilla.org/t/about-common-voice-readme-first/17218)
- [Common Voice on Discord](https://discord.gg/9QTj9zwn)
- [Common Voice on Telegram](https://t.me/mozilla_common_voice)

### Contribute

- [Speak](https://commonvoice.mozilla.org/ta/speak)
- [Write](https://commonvoice.mozilla.org/ta/write)
- [Listen](https://commonvoice.mozilla.org/ta/listen)
- [Review](https://commonvoice.mozilla.org/ta/review)

## Licence

This dataset is released under the [Creative Commons Zero (CC-0)](https://creativecommons.org/public-domain/cc0/) licence. By downloading this data you agree to not determine the identity of speakers in the dataset.
