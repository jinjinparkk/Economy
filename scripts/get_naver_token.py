"""네이버 블로그 OAuth 액세스 토큰 발급 스크립트.

사전 준비:
1. https://developers.naver.com/apps/ 에서 애플리케이션 등록
   - API: "네이버 로그인" + "블로그" 선택
   - Callback URL: http://localhost:5555/callback
2. Client ID / Client Secret 확인

실행:
    py -3 scripts/get_naver_token.py

출력된 NAVER_ACCESS_TOKEN을 .env에 복사하세요.
"""
from __future__ import annotations

import http.server
import sys
import urllib.parse
import webbrowser

import requests

REDIRECT_URI = "http://localhost:5555/callback"
AUTH_URL = "https://nid.naver.com/oauth2.0/authorize"
TOKEN_URL = "https://nid.naver.com/oauth2.0/token"


def main() -> None:
    print("=" * 60)
    print("네이버 블로그 OAuth 토큰 발급")
    print("=" * 60)
    print()
    print("사전 준비:")
    print("  1. https://developers.naver.com/apps/ 에서 앱 등록")
    print("  2. API: '네이버 로그인' + '블로그' 선택")
    print(f"  3. Callback URL: {REDIRECT_URI}")
    print()

    client_id = input("Client ID: ").strip()
    client_secret = input("Client Secret: ").strip()

    if not client_id or not client_secret:
        print("Client ID와 Client Secret을 모두 입력해주세요.")
        sys.exit(1)

    # 1) 인증 URL 열기
    params = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "state": "stock_daily_blog",
    })
    auth_url = f"{AUTH_URL}?{params}"
    print("\n브라우저에서 네이버 로그인을 진행하세요...")
    webbrowser.open(auth_url)

    # 2) 로컬 서버로 callback 수신
    auth_code = None

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal auth_code
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            auth_code = params.get("code", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                "<h1>OK</h1><p>인증이 완료되었습니다. 이 창을 닫아도 됩니다.</p>".encode("utf-8")
            )

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass

    server = http.server.HTTPServer(("localhost", 5555), CallbackHandler)
    print("localhost:5555 에서 callback 대기 중...")
    server.handle_request()
    server.server_close()

    if not auth_code:
        print("인증 코드를 받지 못했습니다.")
        sys.exit(1)

    # 3) 액세스 토큰 교환
    print("\n토큰 교환 중...")
    resp = requests.get(TOKEN_URL, params={
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": auth_code,
        "state": "stock_daily_blog",
    }, timeout=30)

    if resp.status_code != 200:
        print(f"토큰 교환 실패: {resp.status_code} {resp.text}")
        sys.exit(1)

    token_data = resp.json()
    access_token = token_data.get("access_token", "")
    refresh_token = token_data.get("refresh_token", "")
    expires_in = token_data.get("expires_in", "")

    if not access_token:
        error = token_data.get("error_description", token_data.get("error", "unknown"))
        print(f"access_token을 받지 못했습니다: {error}")
        sys.exit(1)

    # 4) 결과 출력
    print()
    print("=" * 60)
    print("아래 값을 .env 파일에 추가하세요:")
    print("=" * 60)
    print(f"NAVER_ACCESS_TOKEN={access_token}")
    print(f"NAVER_REFRESH_TOKEN={refresh_token}")
    print("NAVER_AUTO_PUBLISH=true")
    print("=" * 60)
    print(f"\n토큰 유효기간: {expires_in}초")
    print("만료 시 refresh_token으로 갱신하거나 이 스크립트를 다시 실행하세요.")


if __name__ == "__main__":
    main()
