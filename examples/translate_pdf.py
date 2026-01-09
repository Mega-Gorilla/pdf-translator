#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PDF翻訳サンプルスクリプト

このスクリプトはpdf-translatorの基本的な使い方を示します。
設定変数を変更して、様々な翻訳オプションを試すことができます。

Usage:
    cd examples
    python translate_pdf.py

環境変数（.envファイルから自動読み込み）:
    OPENAI_API_KEY: OpenAI翻訳に必要
    DEEPL_API_KEY: DeepL翻訳に必要
    OPENAI_MODEL: OpenAIモデル指定（デフォルト: gpt-5-nano）
    DISABLE_MODEL_SOURCE_CHECK: PaddleOCRモデルチェックをスキップ（True推奨）
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from pdf_translator.core.side_by_side import SideBySideOrder
    from pdf_translator.translators.base import TranslatorBackend

# プロジェクトルートをパスに追加（開発時用）
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load .env file from project root (API keys, DISABLE_MODEL_SOURCE_CHECK, etc.)
load_dotenv(PROJECT_ROOT / ".env")


# =============================================================================
# 設定変数 - ここを変更して動作をカスタマイズ
# =============================================================================

# 翻訳サービス: "google" | "openai" | "deepl"
# - google: APIキー不要（無料、レート制限あり）
# - openai: OPENAI_API_KEY 環境変数が必要
# - deepl: DEEPL_API_KEY 環境変数が必要
TRANSLATOR = "google"

# 言語設定
SOURCE_LANG = "en"  # 原文の言語
TARGET_LANG = "ja"  # 翻訳先の言語

# デバッグ: レイアウト解析結果のbboxを描画
# True にすると、段落の境界ボックスがPDFに描画されます
DEBUG_DRAW_BBOX = True

# 見開きPDF生成: 原文と翻訳文を左右に並べて表示
SIDE_BY_SIDE = True

# 見開きの配置順序: "original_translated" | "translated_original"
# - original_translated: 左=原文、右=翻訳
# - translated_original: 左=翻訳、右=原文
SIDE_BY_SIDE_ORDER = "original_translated"

# 見開きの間隔（ポイント）
SIDE_BY_SIDE_GAP = 10.0

# レイアウト解析によるカテゴリ分類
# - True: PP-DocLayoutでレイアウト解析を行い、テキスト/タイトルのみ翻訳（推奨）
#         図、表、数式などは自動的にスキップされます
# - False: 全てのテキストを翻訳（数式や図のキャプションも翻訳される可能性あり）
PDF_LAYOUT_ANALYSIS = True

# 厳格モード: 翻訳失敗時の動作
# - False: 失敗したテキストは原文のまま保持（推奨）
# - True: 1つでも失敗したらエラーを発生
STRICT_MODE = False

# 入出力パス
INPUT_PDF = PROJECT_ROOT / "tests" / "fixtures" / "sample_llama.pdf"
INPUT_PDF = PROJECT_ROOT / "tests" / "fixtures" / "sample_autogen_paper.pdf"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# =============================================================================
# メイン処理（通常は変更不要）
# =============================================================================


def get_translator(backend: str) -> TranslatorBackend:
    """翻訳バックエンドを取得する。"""
    if backend == "google":
        from pdf_translator.translators import GoogleTranslator

        return GoogleTranslator()

    elif backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable is not set")
            print("Set it with: export OPENAI_API_KEY='your-api-key'")
            sys.exit(1)

        from pdf_translator.translators import get_openai_translator

        OpenAITranslator = get_openai_translator()
        return OpenAITranslator(api_key=api_key)  # type: ignore[no-any-return]

    elif backend == "deepl":
        api_key = os.environ.get("DEEPL_API_KEY")
        if not api_key:
            print("Error: DEEPL_API_KEY environment variable is not set")
            print("Set it with: export DEEPL_API_KEY='your-api-key'")
            sys.exit(1)

        from pdf_translator.translators import get_deepl_translator

        DeepLTranslator = get_deepl_translator()
        return DeepLTranslator(api_key=api_key)  # type: ignore[no-any-return]

    else:
        print(f"Error: Unknown translator backend: {backend}")
        print("Available options: google, openai, deepl")
        sys.exit(1)


def get_side_by_side_order(order: str) -> SideBySideOrder:
    """見開き順序を取得する。"""
    from pdf_translator.core.side_by_side import SideBySideOrder

    if order == "original_translated":
        return SideBySideOrder.ORIGINAL_TRANSLATED
    elif order == "translated_original":
        return SideBySideOrder.TRANSLATED_ORIGINAL
    else:
        print(f"Error: Unknown side_by_side_order: {order}")
        print("Available options: original_translated, translated_original")
        sys.exit(1)


async def main() -> None:
    """メイン処理。"""
    from pdf_translator.pipeline.translation_pipeline import (
        PipelineConfig,
        TranslationPipeline,
    )

    # 入力ファイル確認
    if not INPUT_PDF.exists():
        print(f"Error: Input PDF not found: {INPUT_PDF}")
        sys.exit(1)

    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 出力ファイル名を生成
    base_name = INPUT_PDF.stem
    suffix = f"_{TRANSLATOR}"
    if DEBUG_DRAW_BBOX:
        suffix += "_bbox"
    if SIDE_BY_SIDE:
        suffix += "_sidebyside"

    output_pdf = OUTPUT_DIR / f"{base_name}{suffix}.pdf"

    # 設定を表示
    print("=" * 60)
    print("PDF Translation Example")
    print("=" * 60)
    print(f"Input:       {INPUT_PDF}")
    print(f"Output:      {output_pdf}")
    print(f"Translator:  {TRANSLATOR}")
    print(f"Languages:   {SOURCE_LANG} -> {TARGET_LANG}")
    print(f"Layout analysis: {PDF_LAYOUT_ANALYSIS}")
    print(f"Debug bbox:  {DEBUG_DRAW_BBOX}")
    print(f"Side-by-side: {SIDE_BY_SIDE}")
    if SIDE_BY_SIDE:
        print(f"  Order:     {SIDE_BY_SIDE_ORDER}")
        print(f"  Gap:       {SIDE_BY_SIDE_GAP} pt")
    print(f"Strict mode: {STRICT_MODE}")
    print("=" * 60)

    # 翻訳バックエンド取得
    print(f"\nInitializing {TRANSLATOR} translator...")
    translator = get_translator(TRANSLATOR)

    # パイプライン設定
    config = PipelineConfig(
        source_lang=SOURCE_LANG,
        target_lang=TARGET_LANG,
        pdf_layout_analysis=PDF_LAYOUT_ANALYSIS,
        debug_draw_bbox=DEBUG_DRAW_BBOX,
        side_by_side=SIDE_BY_SIDE,
        side_by_side_order=get_side_by_side_order(SIDE_BY_SIDE_ORDER),
        side_by_side_gap=SIDE_BY_SIDE_GAP,
        strict_mode=STRICT_MODE,
    )

    # パイプライン実行
    pipeline = TranslationPipeline(translator=translator, config=config)

    print("\nTranslating PDF...")
    result = await pipeline.translate(INPUT_PDF, output_pdf)

    # 結果表示
    print("\n" + "=" * 60)
    print("Translation Complete!")
    print("=" * 60)
    if result.stats:
        print(f"Paragraphs translated: {result.stats['translated_paragraphs']}")
        print(f"Paragraphs skipped:    {result.stats['skipped_paragraphs']}")
    print(f"Output file:           {output_pdf}")
    print(f"File size:             {output_pdf.stat().st_size / 1024:.1f} KB")

    # 見開きPDF確認（パイプラインが自動的に _side_by_side.pdf を生成）
    if SIDE_BY_SIDE:
        sidebyside_path = output_pdf.with_stem(output_pdf.stem + "_side_by_side")
        if sidebyside_path.exists():
            print(f"Side-by-side PDF:      {sidebyside_path}")
            print(f"Side-by-side size:     {sidebyside_path.stat().st_size / 1024:.1f} KB")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
