from voice_assistant.audio import SentenceSplitter, buffered


def test_splits_on_sentence_end_only_with_whitespace():
    splitter = SentenceSplitter()
    out = []
    for token in ["Hallo! Das ", "kostet 3.5 Euro. Und", " z. B. das hier? Ja"]:
        out += splitter.feed(token)
    out += splitter.flush()
    assert out == ["Hallo!", "Das kostet 3.5 Euro.", "Und z. B. das hier?", "Ja"]


def test_newline_is_boundary_and_ordinals_are_not():
    splitter = SentenceSplitter()
    out = splitter.feed("Am 3. Mai geht es los\nDanach mehr. ")
    assert out == ["Am 3. Mai geht es los", "Danach mehr."]


async def test_buffered_passes_items_and_errors():
    async def source():
        yield "a"
        yield "b"
        raise ValueError("kaputt")

    seen = []
    try:
        async for item in buffered(source()):
            seen.append(item)
    except ValueError:
        seen.append("error")
    assert seen == ["a", "b", "error"]
