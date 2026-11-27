from sift.storage.migrations import Migration
from sift.storage.texts import BLOCK

LAYOUT: tuple[Migration, ...] = (
    Migration(
        (
            "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT, WITHOUT ROWID",
            """CREATE TABLE counts(
                id INTEGER PRIMARY KEY CHECK (id = 1),
                documents INTEGER NOT NULL DEFAULT 0 CHECK (documents >= 0),
                chunks INTEGER NOT NULL DEFAULT 0 CHECK (chunks >= 0),
                places INTEGER NOT NULL DEFAULT 0 CHECK (places >= 0),
                tokens INTEGER NOT NULL DEFAULT 0 CHECK (tokens >= 0),
                keyword_changes INTEGER NOT NULL DEFAULT 0,
                vectors INTEGER NOT NULL DEFAULT 0 CHECK (vectors >= 0),
                vector_changes INTEGER NOT NULL DEFAULT 0,
                vectors_written INTEGER NOT NULL DEFAULT 0,
                ordered_at INTEGER NOT NULL DEFAULT 0
            ) STRICT""",
            "INSERT INTO counts(id) VALUES (1)",
            """CREATE TABLE sources(
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                modified_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                folded TEXT NOT NULL
            ) STRICT""",
            "CREATE INDEX sources_folded ON sources(folded)",
            """CREATE TABLE unreadable(
                path TEXT PRIMARY KEY,
                problem TEXT NOT NULL,
                read_again INTEGER NOT NULL CHECK (read_again IN (0, 1)),
                skipped_under TEXT CHECK (skipped_under IN ('off', 'auto')),
                stopped_under TEXT
            ) STRICT, WITHOUT ROWID""",
            """CREATE TABLE documents(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                digest TEXT NOT NULL,
                title TEXT NOT NULL,
                fields TEXT NOT NULL CHECK (json_valid(fields)),
                UNIQUE(source, key)
            ) STRICT""",
            """CREATE TRIGGER documents_added AFTER INSERT ON documents
               BEGIN UPDATE counts SET documents = documents + 1; END""",
            """CREATE TRIGGER documents_deleted AFTER DELETE ON documents
               BEGIN UPDATE counts SET documents = documents - 1; END""",
            f"""CREATE TABLE texts(
                document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                block INTEGER NOT NULL CHECK (block >= 0),
                first_line INTEGER NOT NULL CHECK (first_line >= 1),
                text TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND {BLOCK}),
                PRIMARY KEY(document, block)
            ) STRICT""",
            """CREATE TABLE chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                first_line INTEGER NOT NULL,
                last_line INTEGER NOT NULL,
                start INTEGER NOT NULL,
                length INTEGER NOT NULL,
                place INTEGER NOT NULL
            ) STRICT""",
            "CREATE INDEX chunks_document ON chunks(document)",
            "CREATE UNIQUE INDEX chunks_place ON chunks(place)",
            "CREATE TABLE unembedded(chunk INTEGER PRIMARY KEY) STRICT",
            """CREATE TRIGGER chunks_added AFTER INSERT ON chunks
               BEGIN UPDATE counts SET chunks = chunks + 1, places = max(places, NEW.place);
                     INSERT INTO unembedded(chunk) VALUES (NEW.id); END""",
            """CREATE TRIGGER chunks_deleted AFTER DELETE ON chunks
               BEGIN UPDATE counts SET chunks = chunks - 1;
                     DELETE FROM unembedded WHERE chunk = OLD.id; END""",
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
            """CREATE TABLE keyword_segments(
                id INTEGER PRIMARY KEY,
                first INTEGER NOT NULL,
                last INTEGER NOT NULL CHECK (last >= first),
                chunks INTEGER NOT NULL CHECK (chunks > 0),
                dead INTEGER NOT NULL CHECK (dead BETWEEN 0 AND chunks),
                average REAL NOT NULL CHECK (average >= 0)
            ) STRICT""",
            """CREATE TABLE keyword_packs(
                segment INTEGER NOT NULL REFERENCES keyword_segments(id) ON DELETE CASCADE,
                first TEXT NOT NULL,
                lists BLOB NOT NULL,
                places BLOB NOT NULL,
                PRIMARY KEY(segment, first)
            ) STRICT""",
            """CREATE TABLE keyword_ranges(
                segment INTEGER NOT NULL REFERENCES keyword_segments(id) ON DELETE CASCADE,
                term TEXT NOT NULL,
                chunks INTEGER NOT NULL CHECK (chunks > 0),
                ranges BLOB NOT NULL,
                PRIMARY KEY(segment, term)
            ) STRICT""",
            """CREATE TABLE keyword_blocks(
                segment INTEGER NOT NULL REFERENCES keyword_segments(id) ON DELETE CASCADE,
                term TEXT NOT NULL,
                number INTEGER NOT NULL CHECK (number >= 0),
                postings BLOB NOT NULL,
                positions BLOB NOT NULL,
                PRIMARY KEY(segment, term, number)
            ) STRICT""",
            """CREATE TABLE keyword_lengths(
                segment INTEGER NOT NULL REFERENCES keyword_segments(id) ON DELETE CASCADE,
                block INTEGER NOT NULL,
                lengths BLOB NOT NULL,
                PRIMARY KEY(segment, block)
            ) STRICT""",
            """CREATE TABLE keyword_gone(
                term TEXT NOT NULL,
                segment INTEGER NOT NULL REFERENCES keyword_segments(id) ON DELETE CASCADE,
                chunks INTEGER NOT NULL CHECK (chunks > 0),
                PRIMARY KEY(term, segment)
            ) STRICT, WITHOUT ROWID""",
            "CREATE INDEX keyword_gone_of ON keyword_gone(segment)",
            "CREATE TABLE keyword_dead(place INTEGER PRIMARY KEY) STRICT",
            """CREATE TABLE keyword_beside(
                id INTEGER PRIMARY KEY CHECK (id = 1),
                above INTEGER NOT NULL,
                through INTEGER NOT NULL CHECK (through >= above),
                made_at INTEGER NOT NULL
            ) STRICT""",
            "CREATE TABLE keyword_places(place INTEGER PRIMARY KEY, chunk INTEGER NOT NULL) STRICT",
            """CREATE VIEW keyword_documents AS
               SELECT d.id AS id, d.title AS title,
                      (SELECT group_concat(j.atom, ' ') FROM json_tree(d.fields) j
                       WHERE j.atom IS NOT NULL AND j.fullkey <> '$.title') AS meta
               FROM documents d""",
            """CREATE VIEW keyword_served AS
               SELECT id, first, last, chunks, dead, average FROM keyword_segments
               WHERE last <= coalesce((SELECT above FROM keyword_beside), 9223372036854775807)""",
            """CREATE VIEW keyword_built AS
               SELECT id, first, last, chunks, dead, average FROM keyword_segments
               WHERE first > (SELECT above FROM keyword_beside)""",
            """CREATE TRIGGER keyword_segment_made AFTER INSERT ON keyword_segments
               WHEN NEW.last <= coalesce((SELECT above FROM keyword_beside), 9223372036854775807)
               BEGIN UPDATE counts SET keyword_changes = keyword_changes + 1; END""",
            """CREATE TRIGGER keyword_segment_gone AFTER DELETE ON keyword_segments
               WHEN OLD.last <= coalesce((SELECT above FROM keyword_beside), 9223372036854775807)
               BEGIN UPDATE counts SET keyword_changes = keyword_changes + 1; END""",
            """CREATE TRIGGER keyword_segment_dropped AFTER DELETE ON keyword_segments
               BEGIN DELETE FROM keyword_dead WHERE place BETWEEN OLD.first AND OLD.last; END""",
            """CREATE TABLE vectors(
                chunk INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                vector BLOB NOT NULL
            ) STRICT""",
            """CREATE TABLE codes(
                chunk INTEGER PRIMARY KEY REFERENCES vectors(chunk) ON DELETE CASCADE,
                bits BLOB NOT NULL
            ) STRICT""",
            """CREATE TRIGGER vectors_added AFTER INSERT ON vectors
               BEGIN UPDATE counts SET vectors = vectors + 1, vector_changes = vector_changes + 1,
                                       vectors_written = vectors_written + 1;
                     DELETE FROM unembedded WHERE chunk = NEW.chunk; END""",
            """CREATE TRIGGER vectors_replaced AFTER UPDATE OF vector ON vectors
               BEGIN UPDATE counts SET vector_changes = vector_changes + 1,
                                       vectors_written = vectors_written + 1; END""",
            """CREATE TRIGGER vectors_deleted AFTER DELETE ON vectors
               BEGIN UPDATE counts SET vectors = vectors - 1, vector_changes = vector_changes + 1;
                     INSERT OR IGNORE INTO unembedded(chunk)
                     SELECT OLD.chunk WHERE EXISTS (SELECT 1 FROM chunks WHERE id = OLD.chunk); END""",
        )
    ),
)
