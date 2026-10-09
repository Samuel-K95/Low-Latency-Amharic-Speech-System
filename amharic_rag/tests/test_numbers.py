"""
Unit tests for src/ethiopic_numbers.py.
Covers:
  - Part 1: Ge'ez numeral parsing to int (with ሺ, quirk handling, malformed handling).
  - Part 2: Integer conversion to Amharic spoken words (0..999,999).
  - Part 3: Arabic numeral parsing (integers with commas, percentages, decimals).
"""
import pytest
from src.ethiopic_numbers import geez_to_int, int_to_words, to_words_arabic, int_to_geez

# -----------------------------------------------------------------------------
# Part 1: Ge'ez numeral parsing tests
# -----------------------------------------------------------------------------
def test_geez_to_int_basic_and_tens():
    assert geez_to_int('፩') == 1
    assert geez_to_int('፭') == 5
    assert geez_to_int('፱') == 9
    assert geez_to_int('፲') == 10
    assert geez_to_int('፲፭') == 15
    assert geez_to_int('፳') == 20
    assert geez_to_int('፴') == 30
    assert geez_to_int('፵፭') == 45
    assert geez_to_int('፷') == 60
    assert geez_to_int('፺') == 90

def test_geez_to_int_hundreds():
    assert geez_to_int('፻') == 100
    assert geez_to_int('፩፻') == 100
    assert geez_to_int('፩፻፲') == 110
    assert geez_to_int('፩፻፹') == 180
    assert geez_to_int('፩፻፶፮') == 156
    assert geez_to_int('፪፻') == 200

def test_geez_to_int_thousands_with_shi():
    assert geez_to_int('፩ሺ') == 1000
    assert geez_to_int('፭ሺ') == 5000
    assert geez_to_int('፲ሺ') == 10000
    assert geez_to_int('፴ሺ') == 30000
    assert geez_to_int('፩፻ሺ') == 100000
    assert geez_to_int('፩ሺ፩፻፶፮') == 1156
    assert geez_to_int('፪ሺ፲፩') == 2011
    assert geez_to_int('፲፱፻፺፫') == 1993

def test_geez_to_int_malformed_and_edge_cases():
    # Malformed source typo in Art 156(3)
    assert geez_to_int('፬፵፱') is None
    # Empty or whitespace or invalid strings
    assert geez_to_int('') is None
    assert geez_to_int('   ') is None
    assert geez_to_int('abc') is None
    assert geez_to_int(None) is None
    assert geez_to_int('፩ሺ፪ሺ') is None

# -----------------------------------------------------------------------------
# Part 2: Integer to Amharic words (spoken form) tests
# -----------------------------------------------------------------------------
def test_int_to_words_basic():
    assert int_to_words(0) == 'ዜሮ'
    assert int_to_words(1) == 'አንድ'
    assert int_to_words(10) == 'አስር'
    assert int_to_words(12) == 'አስራ ሁለት'
    assert int_to_words(20) == 'ሃያ'
    assert int_to_words(25) == 'ሃያ አምስት'
    assert int_to_words(45) == 'አርባ አምስት'
    assert int_to_words(60) == 'ስድሳ'
    assert int_to_words(70) == 'ሰባ'
    assert int_to_words(80) == 'ሰማንያ'
    assert int_to_words(90) == 'ዘጠና'

def test_int_to_words_hundreds():
    assert int_to_words(100) == 'አንድ መቶ'
    assert int_to_words(150) == 'አንድ መቶ ሃምሳ'
    assert int_to_words(156) == 'አንድ መቶ ሃምሳ ስድስት'
    assert int_to_words(180) == 'አንድ መቶ ሰማንያ'
    assert int_to_words(200) == 'ሁለት መቶ'
    assert int_to_words(999) == 'ዘጠኝ መቶ ዘጠና ዘጠኝ'

def test_int_to_words_thousands():
    assert int_to_words(1000) == 'አንድ ሺህ'
    assert int_to_words(1156) == 'አንድ ሺህ አንድ መቶ ሃምሳ ስድስት'
    assert int_to_words(2011) == 'ሁለት ሺህ አስራ አንድ'
    assert int_to_words(5000) == 'አምስት ሺህ'
    assert int_to_words(10000) == 'አስር ሺህ'
    assert int_to_words(100000) == 'አንድ መቶ ሺህ'
    assert int_to_words(999999) == 'ዘጠኝ መቶ ዘጠና ዘጠኝ ሺህ ዘጠኝ መቶ ዘጠና ዘጠኝ'

# -----------------------------------------------------------------------------
# Part 3: Arabic digits to Amharic words tests
# -----------------------------------------------------------------------------
def test_to_words_arabic_integers():
    assert to_words_arabic('100') == 'አንድ መቶ'
    assert to_words_arabic('5,000') == 'አምስት ሺህ'
    assert to_words_arabic('100,000') == 'አንድ መቶ ሺህ'

def test_to_words_arabic_percentages():
    assert to_words_arabic('30%') == 'ሠላሳ በመቶ'
    assert to_words_arabic('50%') == 'ሃምሳ በመቶ'
    assert to_words_arabic('100%') == 'አንድ መቶ በመቶ'

def test_to_words_arabic_decimals():
    assert to_words_arabic('1.75') == 'አንድ ነጥብ ሰባ አምስት'
    assert to_words_arabic('1.5') == 'አንድ ነጥብ አምስት'
    assert to_words_arabic('0.5') == 'ዜሮ ነጥብ አምስት'
    assert to_words_arabic('2.25') == 'ሁለት ነጥብ ሃያ አምስት'
