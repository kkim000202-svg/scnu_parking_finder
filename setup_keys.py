"""카카오 키를 .env 파일에 저장한다. 키는 화면에 보이지 않게 입력받는다.

실행: python setup_keys.py
"""

from getpass import getpass
from pathlib import Path

ENV = Path(__file__).resolve().parent / ".env"


def ask(label):
    while True:
        value = getpass(f"{label} 붙여 넣고 Enter (화면에는 안 보여요): ").strip().strip("'\"")
        if len(value) >= 20 and " " not in value:
            return value
        print("  키가 너무 짧거나 띄어쓰기가 있어요. 카카오 개발자 사이트 [앱] → [플랫폼 키]에서 다시 복사해 주세요.")


def main():
    print("카카오 개발자 사이트 → 내 앱 → [앱] → [플랫폼 키] 에서 키를 복사해 오세요.\n")
    js = ask("1) JavaScript 키")
    rest = ask("2) REST API 키")
    if js == rest:
        print("\n두 키가 같아요. JavaScript 키와 REST API 키를 각각 복사했는지 확인하고 다시 실행해 주세요.")
        return
    ENV.write_text(
        "# 카카오 키 (GitHub에 올라가지 않는 파일이에요. 비밀번호처럼 다루세요)\n"
        f"KAKAO_JS_KEY={js}\nKAKAO_REST_KEY={rest}\n", encoding="utf-8")
    print(f"\n저장했어요: {ENV}")
    print(f"JavaScript 키 {js[:4]}…{js[-3:]}, REST API 키 {rest[:4]}…{rest[-3:]}")


if __name__ == "__main__":
    main()
