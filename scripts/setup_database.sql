-- ============================================================
-- One-time database bootstrap (run as the `postgres` superuser)
--
--   psql -U postgres -h localhost -f scripts/setup_database.sql
--
-- Creates the application role + database and enables TimescaleDB.
-- Idempotent: safe to re-run.
-- IMPORTANT: ganti 'change_me' agar sama dengan POSTGRES_PASSWORD di .env
-- ============================================================

-- 1) Application role (login user for the app; not a superuser)
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'btc_analyst') THEN
      CREATE ROLE btc_analyst LOGIN PASSWORD 'change_me';
   END IF;
END
$$;

-- 2) Application database (owned by the app role)
SELECT 'CREATE DATABASE bitcoin_ai OWNER btc_analyst'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'bitcoin_ai') \gexec

-- 3) Enable TimescaleDB inside the application database
--    (extension creation requires superuser, so we do it here once;
--     scripts/init_db.py hanya melakukan CREATE EXTENSION IF NOT EXISTS)
\connect bitcoin_ai
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 4) Make sure the app role owns the public schema objects it will create
GRANT ALL ON SCHEMA public TO btc_analyst;

\echo '=== setup_database.sql selesai: role btc_analyst + db bitcoin_ai + timescaledb siap ==='
