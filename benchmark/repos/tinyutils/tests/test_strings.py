from tinyutils.strings import (
    capitalize_words,
    count_words,
    first_word,
    is_palindrome,
    slugify,
    truncate,
)


def test_truncate_short_text_unchanged():
    assert truncate("hello", 10) == "hello"


def test_truncate_exact_length_unchanged():
    assert truncate("hello", 5) == "hello"


def test_truncate_adds_suffix():
    assert truncate("hello world", 8) == "hello..."


def test_truncate_tiny_limit():
    assert truncate("hello world", 2) == ".."


def test_palindrome():
    assert is_palindrome("A man, a plan, a canal: Panama")
    assert not is_palindrome("hello")
    assert is_palindrome("")


def test_count_words():
    assert count_words("one two  three") == 3
    assert count_words("") == 0


def test_capitalize_words():
    assert capitalize_words("hELLO wORLD") == "Hello World"


def test_slugify():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  --A  B-- ") == "a-b"


def test_first_word():
    assert first_word("  alpha beta") == "alpha"
    assert first_word("   ") == ""
