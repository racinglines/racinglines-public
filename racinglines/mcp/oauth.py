"""
OAuth sign-in for the hosted MCP server, so a connector needs no pasted token: claude.ai, Claude Desktop or Claude
Code adds https://mcp.racinglines.bet/mcp, the browser opens racinglines.bet/mcp/authorize (the normal login first if
needed, then one Allow), and the client holds an access token it refreshes on its own.

The SDK's authorization server (mcp.server.auth: /register, /authorize, /token, /revoke and the .well-known metadata)
calls `Provider`. Nothing is stored per client or per token: client ids, authorization codes, access and refresh
tokens are signed with APP_SECRET, which the web app shares (it signs the code after its Allow page). A token is
accepted while its account is active, not a demo, has a role in RACINGLINES_MCP_ROLES, and its grant counter
(users.prefs["mcp_oauth"]["gen"]) is unchanged; `disconnect()` bumps the counter, ending every OAuth grant of that
account at once. `rl_` tokens (auth.py) go through the same check, so both kinds work on one server.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

from sqlalchemy import text

from racinglines.mcp import auth

MCP_URL = os.environ.get("RACINGLINES_MCP_URL", "https://mcp.racinglines.bet/mcp")
ISSUER = os.environ.get("RACINGLINES_MCP_ISSUER", MCP_URL.rsplit("/mcp", 1)[0])
WEB_URL = os.environ.get("RACINGLINES_WEB_URL", "https://racinglines.bet")
ACCESS_TTL = 3600
REFRESH_TTL = 30 * 86400
CODE_TTL = 300
REQUEST_TTL = 600


def _secret():
    s = os.environ.get("APP_SECRET")
    if not s:
        raise RuntimeError("APP_SECRET is unset: MCP sign-in needs the key the web app and the MCP server share")
    return s.encode()


def enabled():
    return bool(os.environ.get("APP_SECRET"))


def _b64(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign(kind, payload, ttl):
    """`<payload>.<mac>`: payload is base64url JSON with the kind and an expiry; the MAC covers both."""
    body = _b64(json.dumps({**payload, "k": kind, "exp": int(time.time()) + ttl, "n": secrets.token_hex(10)},
                           separators=(",", ":")).encode())
    return body + "." + _b64(hmac.new(_secret(), f"{kind}.{body}".encode(), hashlib.sha256).digest())


def unsign(kind, blob):
    """The payload of a blob this server signed for `kind`, unexpired; None otherwise."""
    if not blob or blob.count(".") != 1 or not enabled():
        return None
    body, mac = blob.split(".")
    want = _b64(hmac.new(_secret(), f"{kind}.{body}".encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(mac, want):
        return None
    try:
        data = json.loads(_unb64(body))
    except ValueError:
        return None
    return data if data.get("k") == kind and data.get("exp", 0) >= time.time() else None


# --- accounts ------------------------------------------------------------------------------------

def account(engine, user_id=None, username=None):
    """dict(id, username, role, gen) for an account allowed MCP access, else None (same rule as auth.lookup)."""
    col, val = ("id", user_id) if user_id is not None else ("username", username)
    with engine.connect() as c:
        r = c.execute(text(f"""SELECT id, username, role, active, coalesce((prefs->'mcp_oauth'->>'gen')::int, 0) AS gen
                               FROM users WHERE {col} = :v"""), dict(v=val)).fetchone()
    if r is None or not r.active or r.username in auth._demo_users() or auth.R.canonical(r.role) not in auth.ROLES:
        return None
    return dict(id=int(r.id), username=r.username, role=auth.R.canonical(r.role), gen=int(r.gen))


def _merge(engine, user_id, patch):
    with engine.begin() as c:
        c.execute(text("""UPDATE users SET prefs = jsonb_set(coalesce(prefs, '{}'::jsonb), '{mcp_oauth}',
                                 coalesce(prefs->'mcp_oauth', '{}'::jsonb) || CAST(:p AS jsonb)) WHERE id = :i"""),
                  dict(p=json.dumps(patch), i=user_id))


def disconnect(engine, user_id):
    """End every OAuth grant of this account (its `rl_` token, if any, is untouched)."""
    with engine.begin() as c:
        n = c.execute(text("""UPDATE users SET prefs = jsonb_set(coalesce(prefs, '{}'::jsonb), '{mcp_oauth}',
                                 coalesce(prefs->'mcp_oauth', '{}'::jsonb) - 'clients'
                                 || jsonb_build_object('gen', coalesce((prefs->'mcp_oauth'->>'gen')::int, 0) + 1))
                              WHERE id = :i"""), dict(i=user_id)).rowcount
    return bool(n)


def _seen(engine, user_id, client_name):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with engine.begin() as c:
        c.execute(text("""UPDATE users SET prefs = jsonb_set(coalesce(prefs, '{}'::jsonb), '{mcp_oauth}',
                                 coalesce(prefs->'mcp_oauth', '{}'::jsonb)
                                 || jsonb_build_object('clients', coalesce(prefs->'mcp_oauth'->'clients', '{}'::jsonb)
                                                                  || jsonb_build_object(CAST(:n AS text), CAST(:t AS text))))
                              WHERE id = :i"""), dict(n=client_name or "MCP client", t=now, i=user_id))


def connections(engine):
    """Every account with MCP access of either kind: [dict(username, role, token_issued_at, clients{name: last})]."""
    with engine.connect() as c:
        rows = c.execute(text("""SELECT username, role, prefs->'mcp'->>'issued_at' AS issued,
                                        prefs->'mcp_oauth'->'clients' AS clients FROM users
                                 WHERE prefs ? 'mcp' OR prefs->'mcp_oauth' ? 'clients' ORDER BY username""")).fetchall()
    return [dict(username=r.username, role=r.role, token_issued_at=r.issued, clients=r.clients or {}) for r in rows]


# --- the web app's half: the Allow page ----------------------------------------------------------

def consent_url(client, params):
    """Where /authorize sends the browser: the web app's Allow page, carrying the signed request."""
    req = sign("req", dict(c=client.client_id, name=client.client_name or "an MCP client",
                           r=str(params.redirect_uri), x=params.redirect_uri_provided_explicitly,
                           ch=params.code_challenge, s=params.scopes or [], st=params.state, rs=params.resource),
               REQUEST_TTL)
    return f"{WEB_URL}/mcp/authorize?" + urlencode(dict(req=req))


def approve(req_blob, user):
    """The web app, after Allow: the client's redirect URL with a fresh code for `user` (dict with id), or None if the
    request is not one this server signed (or it expired)."""
    req = unsign("req", req_blob)
    if req is None:
        return None
    code = sign("code", dict(c=req["c"], r=req["r"], x=req["x"], ch=req["ch"], s=req["s"], rs=req["rs"],
                             u=int(user["id"])), CODE_TTL)
    from mcp.server.auth.provider import construct_redirect_uri
    return construct_redirect_uri(req["r"], code=code, state=req["st"])


def deny_url(req_blob):
    req = unsign("req", req_blob)
    if req is None:
        return None
    from mcp.server.auth.provider import construct_redirect_uri
    return construct_redirect_uri(req["r"], error="access_denied", state=req["st"])


def request_info(req_blob):
    """What the Allow page shows: dict(client name, redirect host) or None."""
    req = unsign("req", req_blob)
    if req is None:
        return None
    from urllib.parse import urlparse
    return dict(client=req["name"], redirect_host=urlparse(req["r"]).netloc)


# --- the MCP server's half: the SDK provider -----------------------------------------------------

def provider(engine_fn):
    """The provider object (the SDK is imported here, so the web app can use the helpers above without it)."""
    from mcp.server.auth.provider import (
        AccessToken,
        AuthorizationCode,
        AuthorizeError,
        RefreshToken,
        TokenError,
    )
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    used_codes = {}     # code nonce -> expiry: a code works once (one server process)

    class Provider:
        async def register_client(self, client_info):
            # The SDK made a random client_id; replace it with a signed one that carries the metadata, so get_client
            # needs no table. The SDK returns this same object to the client.
            meta = client_info.model_dump(mode="json", exclude={"client_id", "client_secret", "client_id_issued_at",
                                                                "client_secret_expires_at"}, exclude_none=True)
            client_info.client_id = sign("client", dict(m=meta), 10 * 365 * 86400)
            if client_info.client_secret is not None:
                client_info.client_secret = self._client_secret(client_info.client_id)

        @staticmethod
        def _client_secret(client_id):
            return _b64(hmac.new(_secret(), f"secret.{client_id}".encode(), hashlib.sha256).digest())

        async def get_client(self, client_id):
            data = unsign("client", client_id)
            if data is None:
                return None
            meta = data["m"]
            secret = None if meta.get("token_endpoint_auth_method") == "none" else self._client_secret(client_id)
            return OAuthClientInformationFull(client_id=client_id, client_secret=secret, **meta)

        async def authorize(self, client, params):
            if not enabled():
                raise AuthorizeError("server_error", "sign-in is not configured on this server (APP_SECRET)")
            return consent_url(client, params)

        async def load_authorization_code(self, client, authorization_code):
            d = unsign("code", authorization_code)
            if d is None or d["c"] != client.client_id or used_codes.get(d["n"]):
                return None
            return AuthorizationCode(code=authorization_code, scopes=d["s"], expires_at=d["exp"], client_id=d["c"],
                                     code_challenge=d["ch"], redirect_uri=d["r"], redirect_uri_provided_explicitly=d["x"],
                                     resource=d["rs"], subject=str(d["u"]))

        async def exchange_authorization_code(self, client, authorization_code):
            d = unsign("code", authorization_code.code)
            now = time.time()
            for k in [k for k, exp in used_codes.items() if exp < now]:
                del used_codes[k]
            if d is None or used_codes.get(d["n"]):
                raise TokenError("invalid_grant", "authorization code is invalid or already used")
            used_codes[d["n"]] = d["exp"]
            return self._tokens(client, d["u"], d["s"], d["rs"])

        def _tokens(self, client, user_id, scopes, resource):
            eng = engine_fn()
            who = account(eng, user_id=user_id)
            if who is None:
                raise TokenError("invalid_grant", "this account may not use the MCP server")
            _seen(eng, who["id"], client.client_name)
            common = dict(u=who["id"], g=who["gen"], s=scopes, rs=resource)
            return OAuthToken(access_token=sign("at", dict(common, c=client.client_id), ACCESS_TTL),
                              expires_in=ACCESS_TTL, scope=" ".join(scopes) or None,
                              refresh_token=sign("rt", dict(common, c=client.client_id), REFRESH_TTL))

        async def load_refresh_token(self, client, refresh_token):
            d = unsign("rt", refresh_token)
            if d is None or d["c"] != client.client_id:
                return None
            who = account(engine_fn(), user_id=d["u"])
            if who is None or who["gen"] != d["g"]:
                return None
            return RefreshToken(token=refresh_token, client_id=d["c"], scopes=d["s"], expires_at=d["exp"],
                                resource=d["rs"], subject=str(d["u"]))

        async def exchange_refresh_token(self, client, refresh_token, scopes):
            return self._tokens(client, int(refresh_token.subject), scopes or refresh_token.scopes, refresh_token.resource)

        async def load_access_token(self, token):
            eng = engine_fn()
            if token.startswith(auth.PREFIX):
                who = auth.lookup(eng, token)
                if who is None:
                    return None
                return AccessToken(token=token, client_id="rl_token", scopes=[], subject=str(who["id"]),
                                   claims=dict(username=who["username"], role=who["role"]))
            d = unsign("at", token)
            if d is None:
                return None
            who = account(eng, user_id=d["u"])
            if who is None or who["gen"] != d["g"]:
                return None
            return AccessToken(token=token, client_id=d["c"], scopes=d["s"], expires_at=d["exp"], resource=d["rs"],
                               subject=str(who["id"]), claims=dict(username=who["username"], role=who["role"]))

        async def revoke_token(self, token):
            if token.subject:
                disconnect(engine_fn(), int(token.subject))

    return Provider()


def auth_settings():
    from mcp.server.auth.settings import (
        AuthSettings,
        ClientRegistrationOptions,
        RevocationOptions,
    )
    return AuthSettings(issuer_url=ISSUER, resource_server_url=MCP_URL, validate_token_resource=False,
                        service_documentation_url=f"{WEB_URL}/docs/mcp/",
                        client_registration_options=ClientRegistrationOptions(enabled=True),
                        revocation_options=RevocationOptions(enabled=True))


def warn_if_disabled():
    if not enabled():
        print("racinglines mcp --http: APP_SECRET is unset, so OAuth sign-in (claude.ai, Claude Desktop) is off; "
              "rl_ tokens still work.", file=sys.stderr)
