"""
LDAP Inspection Tool
--------------------
Run this inside Docker to browse your AD structure:

    docker exec -it <web-container> python ldap_inspect.py

Or run with arguments:
    python ldap_inspect.py users          # list all users
    python ldap_inspect.py groups         # list all groups
    python ldap_inspect.py user sokha     # inspect a specific user
    python ldap_inspect.py ous            # list all OUs (folder structure)
"""

import sys
import os
import ldap

# ── Config (reads from env — all values must be set in .env.dev) ──────────────
SERVER   = os.environ.get("AUTH_LDAP_SERVER_URI")
BIND_DN  = os.environ.get("AUTH_LDAP_BIND_DN")
PASSWORD = os.environ.get("AUTH_LDAP_BIND_PASSWORD")
BASE_DN  = os.environ.get("AUTH_LDAP_USER_BASE_DN")


def connect():
    conn = ldap.initialize(SERVER)
    conn.set_option(ldap.OPT_REFERRALS, 0)
    conn.set_option(ldap.OPT_PROTOCOL_VERSION, ldap.VERSION3)
    conn.set_option(ldap.OPT_NETWORK_TIMEOUT, 5.0)
    conn.simple_bind_s(BIND_DN, PASSWORD)
    print(f"✅ Connected to {SERVER}\n")
    return conn


def list_ous(conn):
    """Show the folder/OU structure of your AD — helps you find where groups live."""
    print("=" * 60)
    print("📁 OU STRUCTURE")
    print("=" * 60)
    results = conn.search_s(
        BASE_DN,
        ldap.SCOPE_SUBTREE,
        "(objectClass=organizationalUnit)",
        ["ou", "distinguishedName"]
    )
    for dn, attrs in results:
        ou = attrs.get("ou", [b""])[0].decode()
        print(f"  📁 {ou}")
        print(f"     DN: {dn}\n")


def list_groups(conn, limit=50):
    """List groups — shows CN and full DN (use DN in your .env settings)."""
    print("=" * 60)
    print(f"👥 GROUPS (first {limit})")
    print("=" * 60)
    results = conn.search_s(
        BASE_DN,
        ldap.SCOPE_SUBTREE,
        "(objectClass=group)",
        ["cn", "distinguishedName", "member"]
    )
    for i, (dn, attrs) in enumerate(results[:limit]):
        cn      = attrs.get("cn",     [b""])[0].decode()
        members = attrs.get("member", [])
        print(f"  👥 {cn}")
        print(f"     DN:      {dn}")
        print(f"     Members: {len(members)}\n")
    print(f"  ... total groups found: {len(results)}")


def list_users(conn, limit=20):
    """List users with their group memberships."""
    print("=" * 60)
    print(f"👤 USERS (first {limit})")
    print("=" * 60)
    results = conn.search_s(
        BASE_DN,
        ldap.SCOPE_SUBTREE,
        "(&(objectClass=user)(objectCategory=person))",
        ["sAMAccountName", "givenName", "sn", "mail", "memberOf"]
    )
    for dn, attrs in results[:limit]:
        username = attrs.get("sAMAccountName", [b""])[0].decode()
        first    = attrs.get("givenName",       [b""])[0].decode()
        last     = attrs.get("sn",              [b""])[0].decode()
        email    = attrs.get("mail",            [b""])[0].decode()
        groups   = [g.decode().split(",")[0].replace("CN=","") for g in attrs.get("memberOf", [])]
        print(f"  👤 {username} ({first} {last})")
        print(f"     Email:  {email}")
        print(f"     Groups: {', '.join(groups) if groups else 'none'}\n")
    print(f"  ... total users found: {len(results)}")


def inspect_user(conn, username):
    """Show all attributes of a specific user."""
    print("=" * 60)
    print(f"🔍 USER DETAIL: {username}")
    print("=" * 60)
    results = conn.search_s(
        BASE_DN,
        ldap.SCOPE_SUBTREE,
        f"(|(sAMAccountName={username})(mail={username}))",
    )
    if not results:
        print(f"  ❌ User '{username}' not found.")
        return
    dn, attrs = results[0]
    print(f"  DN: {dn}\n")
    for key, values in sorted(attrs.items()):
        for v in values:
            try:
                print(f"  {key}: {v.decode()}")
            except Exception:
                print(f"  {key}: (binary)")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        conn = connect()
    except ldap.LDAPError as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "ous"

    if cmd == "ous":
        list_ous(conn)
    elif cmd == "groups":
        list_groups(conn)
    elif cmd == "users":
        list_users(conn)
    elif cmd == "user" and len(sys.argv) > 2:
        inspect_user(conn, sys.argv[2])
    else:
        print(__doc__)

    conn.unbind_s()
