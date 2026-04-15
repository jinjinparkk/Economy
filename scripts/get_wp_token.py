"""WordPress.com OAuth2 액세스 토큰 발급 스크립트.

사전 준비:
1. https://developer.wordpress.com/apps/ 에서 앱 등록
2. Redirect URL을 http://localhost:9876/callback 으로 설정
3. Client ID와 Client Secret을 아래 프롬프트에 입력

실행:
    py -3 scripts/get_wp_token.py

출력된 WP_ACCESS_TOKEN, WP_SITE_ID를 .env에 복사하세요.
"""
from __future__ import annotations

import http.server
import json
import sys
import urllib.parse
import webbrowser

import requests

REDIRECT_URI = "http://localhost:9876/callback"
TOKEN_URL = "https://public-api.wordpress.com/oauth2/token"
AUTH_URL = "https://public-api.wordpress.com/oauth2/authorize"
ME_URL = "https://public-api.wordpress.com/rest/v1.1/me"
SITES_URL = "https://public-api.wordpress.com/rest/v1.1/me/sites"


def main() -> None:
    print("=" * 60)
    print("WordPress.com OAuth2 토큰 발급")
    print("=" * 60)
    print()
    print("사전 준비:")
    print("  1. https://developer.wordpress.com/apps/ 에서 앱 등록")
    print(f"  2. Redirect URL: {REDIRECT_URI}")
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
        "scope": "global",
    })
    auth_url = f"{AUTH_URL}?{params}"
    print(f"\n브라우저에서 인증을 진행하세요...")
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
            self.wfile.write(b"<h1>OK</h1><p>This window can be closed.</p>")

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass  # 콘솔 로그 억제

    server = http.server.HTTPServer(("localhost", 9876), CallbackHandler)
    print("localhost:9876 에서 callback 대기 중...")
    server.handle_request()
    server.server_close()

    if not auth_code:
        print("인증 코드를 받지 못했습니다.")
        sys.exit(1)

    # 3) 액세스 토큰 교환
    print("\n토큰 교환 중...")
    resp = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "code": auth_code,
        "grant_type": "authorization_code",
    }, timeout=30)

    if resp.status_code != 200:
        print(f"토큰 교환 실패: {resp.status_code} {resp.text}")
        sys.exit(1)

    token_data = resp.json()
    access_token = token_data.get("access_token", "")
    blog_id = token_data.get("blog_id", "")

    if not access_token:
        print("access_token을 받지 못했습니다.")
        sys.exit(1)

    # 4) 사이트 정보 조회
    print("\n사이트 정보 조회 중...")
    headers = {"Authorization": f"Bearer {access_token}"}
    sites_resp = requests.get(SITES_URL, headers=headers, timeout=30)
    if sites_resp.status_code == 200:
        sites = sites_resp.json().get("sites", [])
        if sites:
            print("\n사용 가능한 사이트:")
            for s in sites:
                print(f"  - ID: {s['ID']}  URL: {s['URL']}  이름: {s['name']}")
            if not blog_id:
                blog_id = str(sites[0]["ID"])

    # 5) 결과 출력
    print()
    print("=" * 60)
    print("아래 값을 .env 파일에 추가하세요:")
    print("=" * 60)
    print(f"WP_ACCESS_TOKEN={access_token}")
    print(f"WP_SITE_ID={blog_id}")
    print("WP_AUTO_PUBLISH=true")
    print("=" * 60)


if __name__ == "__main__":
    main()
