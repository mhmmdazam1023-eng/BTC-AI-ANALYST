-- ============================================================
-- PERBAIKAN HAK AKSES (jalankan sebagai superuser `postgres`)
--
--   psql -U postgres -h localhost -d bitcoin_ai -f scripts/fix_permissions.sql
--
-- Idempotent: aman dijalankan berulang.
--
-- MASALAH YANG DIPERBAIKI
-- -----------------------
-- Gejala: `permission denied for table macro_indicators` (atau tabel lain)
-- saat menjalankan run_daily.py / run_collectors.py.
--
-- Penyebab: sebagian tabel dimiliki oleh role `postgres`, bukan
-- `btc_analyst`. `setup_database.sql` hanya memberi GRANT pada SKEMA
-- (`GRANT ALL ON SCHEMA public`) — itu memperbolehkan role membuat objek
-- baru, TAPI tidak memberi hak apa pun atas tabel yang sudah dibuat oleh
-- role lain. Begitu sebuah tabel dimiliki `postgres`, `btc_analyst`
-- kehilangan akses sepenuhnya sampai diberi GRANT eksplisit.
--
-- Skrip ini melakukan dua hal:
--   1. GRANT eksplisit atas semua tabel & sequence yang ada sekarang.
--   2. PINDAH KEPEMILIKAN ke `btc_analyst` supaya masalah yang sama tidak
--      terulang — aplikasi seharusnya memiliki objeknya sendiri.
--   3. ALTER DEFAULT PRIVILEGES supaya objek yang dibuat `postgres` di
--      masa depan tetap bisa diakses.
--
-- Catatan TimescaleDB: GRANT dan perubahan owner pada hypertable otomatis
-- merambat ke chunk-nya di TimescaleDB 2.x, jadi skema `_timescaledb_*`
-- TIDAK perlu (dan tidak boleh) disentuh manual.
-- ============================================================

\echo '=== 1) Hak akses skema ==='
GRANT ALL ON SCHEMA public TO btc_analyst;

\echo '=== 2) GRANT atas tabel & sequence yang ada ==='
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO btc_analyst;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO btc_analyst;

\echo '=== 3) Pindahkan kepemilikan ke btc_analyst (agar tidak terulang) ==='
DO $$
DECLARE
    obj record;
    moved integer := 0;
BEGIN
    FOR obj IN
        SELECT c.oid::regclass AS ident, c.relkind
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p', 'v', 'm', 'S')
          AND pg_get_userbyid(c.relowner) <> 'btc_analyst'
    LOOP
        IF obj.relkind = 'S' THEN
            EXECUTE format('ALTER SEQUENCE %s OWNER TO btc_analyst', obj.ident);
        ELSIF obj.relkind = 'v' THEN
            EXECUTE format('ALTER VIEW %s OWNER TO btc_analyst', obj.ident);
        ELSIF obj.relkind = 'm' THEN
            EXECUTE format('ALTER MATERIALIZED VIEW %s OWNER TO btc_analyst', obj.ident);
        ELSE
            EXECUTE format('ALTER TABLE %s OWNER TO btc_analyst', obj.ident);
        END IF;
        moved := moved + 1;
        RAISE NOTICE 'owner -> btc_analyst: %', obj.ident;
    END LOOP;
    RAISE NOTICE '% objek dipindahkan kepemilikannya', moved;
END
$$;

\echo '=== 4) Default privileges untuk objek yang dibuat postgres nanti ==='
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON TABLES TO btc_analyst;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON SEQUENCES TO btc_analyst;

\echo '=== 5) Verifikasi: harus TIDAK ADA baris di bawah ini ==='
SELECT c.relname AS tabel_tanpa_akses,
       pg_get_userbyid(c.relowner) AS owner
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
  AND c.relkind = 'r'
  AND NOT (
      has_table_privilege('btc_analyst', c.oid, 'SELECT')
      AND has_table_privilege('btc_analyst', c.oid, 'INSERT')
  )
ORDER BY 1;

\echo '=== fix_permissions.sql selesai ==='
\echo 'Verifikasi dari aplikasi:  .venv\\Scripts\\python scripts\\check_db.py'
