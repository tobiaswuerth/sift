from sift.storage.documents import folded_names
from sift.storage.migrations import Migration
from sift.storage.texts import BLOCK, texts_cut

LAYOUT: tuple[Migration, ...] = (
    (
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT, WITHOUT ROWID",
        """CREATE TABLE sources(
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            modified_ns INTEGER NOT NULL,
            size INTEGER NOT NULL,
            name TEXT NOT NULL GENERATED ALWAYS AS (substr(path, length(rtrim(path, replace(path, '/', ''))) + 1))
        ) STRICT""",
        "CREATE INDEX sources_name ON sources(name COLLATE NOCASE)",
        """CREATE TABLE unreadable(
            path TEXT PRIMARY KEY,
            problem TEXT NOT NULL,
            read_again INTEGER NOT NULL CHECK (read_again IN (0, 1))
        ) STRICT, WITHOUT ROWID""",
        """CREATE TABLE documents(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            digest TEXT NOT NULL,
            title TEXT NOT NULL,
            fields TEXT NOT NULL CHECK (json_valid(fields)),
            text TEXT NOT NULL,
            UNIQUE(source, key)
        ) STRICT""",
        """CREATE TABLE chunks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            first_line INTEGER NOT NULL,
            last_line INTEGER NOT NULL,
            start INTEGER NOT NULL,
            length INTEGER NOT NULL
        ) STRICT""",
        "CREATE INDEX chunks_document ON chunks(document)",
        """CREATE TABLE links(
            document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            field TEXT,
            line INTEGER NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('wikilink', 'embed', 'markdown')),
            target TEXT NOT NULL,
            key TEXT NOT NULL,
            fragment TEXT,
            shown TEXT,
            PRIMARY KEY(document, ordinal)
        ) STRICT, WITHOUT ROWID""",
        "CREATE INDEX links_key ON links(key)",
        """CREATE TABLE names(
            key TEXT NOT NULL,
            document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('name', 'alias')),
            PRIMARY KEY(key, document, kind)
        ) STRICT, WITHOUT ROWID""",
        "CREATE INDEX names_document ON names(document)",
        """CREATE TABLE counts(
            id INTEGER PRIMARY KEY CHECK (id = 1),
            documents INTEGER NOT NULL CHECK (documents >= 0),
            chunks INTEGER NOT NULL CHECK (chunks >= 0)
        ) STRICT""",
        "INSERT INTO counts(id, documents, chunks) VALUES (1, 0, 0)",
        """CREATE TRIGGER documents_added AFTER INSERT ON documents
           BEGIN UPDATE counts SET documents = documents + 1; END""",
        """CREATE TRIGGER documents_deleted AFTER DELETE ON documents
           BEGIN UPDATE counts SET documents = documents - 1; END""",
        """CREATE TRIGGER chunks_added AFTER INSERT ON chunks
           BEGIN UPDATE counts SET chunks = chunks + 1; END""",
        """CREATE TRIGGER chunks_deleted AFTER DELETE ON chunks
           BEGIN UPDATE counts SET chunks = chunks - 1; END""",
    ),
    (
        "ALTER TABLE counts ADD COLUMN tokens INTEGER NOT NULL DEFAULT 0 CHECK (tokens >= 0)",
        """CREATE TABLE keyword_segments(
            id INTEGER PRIMARY KEY,
            first INTEGER NOT NULL,
            last INTEGER NOT NULL CHECK (last >= first),
            chunks INTEGER NOT NULL CHECK (chunks > 0),
            dead INTEGER NOT NULL CHECK (dead BETWEEN 0 AND chunks),
            tokens INTEGER NOT NULL CHECK (tokens >= 0),
            average REAL NOT NULL CHECK (average >= 0)
        ) STRICT""",
        """CREATE TABLE keyword_postings(
            segment INTEGER NOT NULL REFERENCES keyword_segments(id) ON DELETE CASCADE,
            term TEXT NOT NULL,
            chunks INTEGER NOT NULL CHECK (chunks > 0),
            postings BLOB,
            positions BLOB,
            ranges BLOB,
            CHECK ((postings IS NULL) = (positions IS NULL) AND (postings IS NULL) <> (ranges IS NULL)),
            PRIMARY KEY(segment, term)
        ) STRICT, WITHOUT ROWID""",
        """CREATE TABLE keyword_blocks(
            segment INTEGER NOT NULL,
            term TEXT NOT NULL,
            number INTEGER NOT NULL CHECK (number >= 0),
            postings BLOB NOT NULL,
            positions BLOB NOT NULL,
            PRIMARY KEY(segment, term, number),
            FOREIGN KEY(segment, term) REFERENCES keyword_postings(segment, term) ON DELETE CASCADE
        ) STRICT, WITHOUT ROWID""",
        """CREATE TABLE keyword_lengths(
            segment INTEGER NOT NULL REFERENCES keyword_segments(id) ON DELETE CASCADE,
            block INTEGER NOT NULL,
            lengths BLOB NOT NULL,
            PRIMARY KEY(segment, block)
        ) STRICT, WITHOUT ROWID""",
        """CREATE TABLE keyword_terms(
            term TEXT PRIMARY KEY,
            chunks INTEGER NOT NULL CHECK (chunks > 0)
        ) STRICT, WITHOUT ROWID""",
        "CREATE TABLE keyword_dead(chunk INTEGER PRIMARY KEY) STRICT",
        """CREATE VIEW keyword_documents AS
           SELECT d.id AS id, d.title AS title, d.text AS text,
                  (SELECT group_concat(j.atom, ' ') FROM json_tree(d.fields) j
                   WHERE j.atom IS NOT NULL AND j.fullkey <> '$.title') AS meta
           FROM documents d""",
        """CREATE TRIGGER keyword_segment_dropped AFTER DELETE ON keyword_segments
           BEGIN DELETE FROM keyword_dead WHERE chunk BETWEEN OLD.first AND OLD.last; END""",
    ),
    (
        "ALTER TABLE counts ADD COLUMN vectors INTEGER NOT NULL DEFAULT 0 CHECK (vectors >= 0)",
        "ALTER TABLE counts ADD COLUMN vector_changes INTEGER NOT NULL DEFAULT 0",
        """CREATE TABLE vectors(
            chunk INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            vector BLOB NOT NULL
        ) STRICT""",
        """CREATE TABLE codes(
            chunk INTEGER PRIMARY KEY REFERENCES vectors(chunk) ON DELETE CASCADE,
            bits BLOB NOT NULL
        ) STRICT""",
        """CREATE TRIGGER vectors_added AFTER INSERT ON vectors
           BEGIN UPDATE counts SET vectors = vectors + 1, vector_changes = vector_changes + 1; END""",
        """CREATE TRIGGER vectors_deleted AFTER DELETE ON vectors
           BEGIN UPDATE counts SET vectors = vectors - 1, vector_changes = vector_changes + 1; END""",
        """CREATE TRIGGER vectors_replaced AFTER UPDATE OF vector ON vectors
           BEGIN UPDATE counts SET vector_changes = vector_changes + 1; END""",
    ),
    (
        "CREATE TABLE unembedded(chunk INTEGER PRIMARY KEY) STRICT",
        "INSERT INTO unembedded(chunk) SELECT id FROM chunks WHERE id NOT IN (SELECT chunk FROM vectors)",
        """CREATE TRIGGER chunks_unembedded AFTER INSERT ON chunks
           BEGIN INSERT INTO unembedded(chunk) VALUES (NEW.id); END""",
        """CREATE TRIGGER chunks_gone_unembedded AFTER DELETE ON chunks
           BEGIN DELETE FROM unembedded WHERE chunk = OLD.id; END""",
        """CREATE TRIGGER vectors_embedded AFTER INSERT ON vectors
           BEGIN DELETE FROM unembedded WHERE chunk = NEW.chunk; END""",
        """CREATE TRIGGER vectors_forgotten AFTER DELETE ON vectors
           WHEN EXISTS (SELECT 1 FROM chunks WHERE id = OLD.chunk)
           BEGIN INSERT OR IGNORE INTO unembedded(chunk) VALUES (OLD.chunk); END""",
    ),
    ("ALTER TABLE keyword_segments DROP COLUMN tokens",),
    (
        f"""CREATE TABLE texts(
            document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            block INTEGER NOT NULL CHECK (block >= 0),
            first_line INTEGER NOT NULL CHECK (first_line >= 1),
            text TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND {BLOCK}),
            PRIMARY KEY(document, block)
        ) STRICT""",
        texts_cut,
        "DROP VIEW keyword_documents",
        "ALTER TABLE documents DROP COLUMN text",
        """CREATE VIEW keyword_documents AS
           SELECT d.id AS id, d.title AS title,
                  (SELECT group_concat(j.atom, ' ') FROM json_tree(d.fields) j
                   WHERE j.atom IS NOT NULL AND j.fullkey <> '$.title') AS meta
           FROM documents d""",
    ),
    ("ALTER TABLE unreadable ADD COLUMN skipped_under TEXT CHECK (skipped_under IN ('off', 'auto'))",),
    (
        "ALTER TABLE chunks ADD COLUMN place INTEGER",
        "UPDATE chunks SET place = id",
        "CREATE UNIQUE INDEX chunks_place ON chunks(place)",
        "ALTER TABLE counts ADD COLUMN places INTEGER NOT NULL DEFAULT 0 CHECK (places >= 0)",
        "UPDATE counts SET places = coalesce((SELECT seq FROM sqlite_sequence WHERE name = 'chunks'), 0)",
        """CREATE TRIGGER chunks_placed AFTER INSERT ON chunks WHEN NEW.place IS NULL
           BEGIN UPDATE counts SET places = max(places + 1, NEW.id);
                 UPDATE chunks SET place = (SELECT places FROM counts) WHERE id = NEW.id; END""",
        """CREATE TRIGGER chunks_given_places AFTER INSERT ON chunks WHEN NEW.place IS NOT NULL
           BEGIN UPDATE counts SET places = max(places, NEW.place); END""",
        "ALTER TABLE keyword_dead RENAME COLUMN chunk TO place",
        "ALTER TABLE counts ADD COLUMN vectors_written INTEGER NOT NULL DEFAULT 0",
        "UPDATE counts SET vectors_written = vectors",
        """CREATE TRIGGER vectors_written AFTER INSERT ON vectors
           BEGIN UPDATE counts SET vectors_written = vectors_written + 1; END""",
        """CREATE TRIGGER vectors_rewritten AFTER UPDATE OF vector ON vectors
           BEGIN UPDATE counts SET vectors_written = vectors_written + 1; END""",
        "ALTER TABLE counts ADD COLUMN ordered_at INTEGER NOT NULL DEFAULT 0",
    ),
    ("ALTER TABLE unreadable ADD COLUMN stopped_under TEXT",),
    (
        "ALTER TABLE sources ADD COLUMN folded TEXT NOT NULL DEFAULT ''",
        folded_names,
        "DROP INDEX sources_name",
        "CREATE INDEX sources_folded ON sources(folded)",
    ),
    (
        "ALTER TABLE counts ADD COLUMN keyword_changes INTEGER NOT NULL DEFAULT 0",
        """CREATE TRIGGER keyword_segment_made AFTER INSERT ON keyword_segments
           BEGIN UPDATE counts SET keyword_changes = keyword_changes + 1; END""",
        """CREATE TRIGGER keyword_segment_gone AFTER DELETE ON keyword_segments
           BEGIN UPDATE counts SET keyword_changes = keyword_changes + 1; END""",
    ),
    (
        """CREATE TABLE keyword_beside(
            id INTEGER PRIMARY KEY CHECK (id = 1),
            above INTEGER NOT NULL,
            through INTEGER NOT NULL CHECK (through >= above),
            made_at INTEGER NOT NULL
        ) STRICT""",
        "CREATE TABLE keyword_places(place INTEGER PRIMARY KEY, chunk INTEGER NOT NULL) STRICT",
        """CREATE VIEW keyword_served AS
           SELECT id, first, last, chunks, dead, average FROM keyword_segments
           WHERE last <= coalesce((SELECT above FROM keyword_beside), 9223372036854775807)""",
        """CREATE VIEW keyword_built AS
           SELECT id, first, last, chunks, dead, average FROM keyword_segments
           WHERE first > (SELECT above FROM keyword_beside)""",
        "DROP TRIGGER keyword_segment_made",
        "DROP TRIGGER keyword_segment_gone",
        """CREATE TRIGGER keyword_segment_made AFTER INSERT ON keyword_segments
           WHEN NEW.last <= coalesce((SELECT above FROM keyword_beside), 9223372036854775807)
           BEGIN UPDATE counts SET keyword_changes = keyword_changes + 1; END""",
        """CREATE TRIGGER keyword_segment_gone AFTER DELETE ON keyword_segments
           WHEN OLD.last <= coalesce((SELECT above FROM keyword_beside), 9223372036854775807)
           BEGIN UPDATE counts SET keyword_changes = keyword_changes + 1; END""",
    ),
)
