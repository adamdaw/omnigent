---
name: voice
description: Add a short spoken section to replies while Adam is using voice. Use when he says "voice on"; stop when he says "voice off".
---

# voice — a spoken section for the polly-voice client

The `polly-voice` client reads one section of your reply aloud. The full reply
stays on screen as the record, so the spoken section only has to tell Adam what
happened and what you need from him.

Voice stays on for this session until Adam says "voice off". Confirm each change
in one short line.

## Format

When a reply reports an outcome, a blocker, or a decision for Adam, end it with
this section, using exactly this heading on its own line:

    ### Spoken

    <the spoken text>

Nothing may follow the spoken text. Use the heading once per reply, and never
use it for anything else. The client ignores any reply without it.

## Spoken text

- Say the outcome first, then any blocker, then the next decision.
- If you asked Adam a question, repeat it as the last sentence.
- Use one to three plain sentences and no more than 60 words.
- Write for the ear: full sentences, with no lists, Markdown, tables or emoji.
- Sound like speech, not a status report. Use contractions and concrete verbs,
  and keep one conversational thought per sentence. Avoid noun chains such as
  "the spec now records the first live run".
- Fold a short recommendation into the sentence before it rather than leaving
  it alone. Write "Should we wait a few days? I'd recommend that." as "I'd
  suggest waiting a few days before we build it."
- Don't repeat the same words across sentences.
- Leave out code, commands, file paths, URLs, commit hashes and long numbers.
  Name the thing in ordinary words, and say "details on screen" if Adam needs them.
- Say tool and vendor names as words, and spell out unusual abbreviations.

## When to stay silent

Leave the section out when a reply only reports progress, such as a dispatch or
a wait on a sub-agent. The working sound tells Adam that you are busy, so
spoken updates would only interrupt him.

## Safety

- Spoken text never approves anything. Adam approves at the keyboard.
- Do not add this section in a Vacatia or other work session. Tell Adam instead.
