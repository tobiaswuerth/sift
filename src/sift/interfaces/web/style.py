STYLE = """
:root { color-scheme: light dark; --line: #8884; --mark: #fd4a; --muted: #888; --accent: #36c; }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.45 system-ui, sans-serif; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header { position: sticky; top: 0; padding: .6rem 1rem; border-bottom: 1px solid var(--line); background: Canvas; }
form.search { display: flex; gap: .5rem; align-items: center; }
form.search .home { font-weight: 700; margin-right: .5rem; }
form.search input[type=search] { flex: 1; font: inherit; padding: .35rem .5rem; }
form.search select, form.search button { font: inherit; padding: .3rem .5rem; }
main { display: grid; grid-template-columns: 15rem minmax(18rem, 1fr) minmax(20rem, 1.3fr); gap: 1rem; padding: 1rem; }
aside h3 { font-size: .8rem; text-transform: uppercase; color: var(--muted); margin: 1rem 0 .3rem; }
aside ul { list-style: none; margin: 0; padding: 0; }
aside li { display: flex; justify-content: space-between; gap: .5rem; }
aside li.on a { font-weight: 700; }
.count { color: var(--muted); font-size: .85rem; }
.chips { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .6rem; }
.chip { border: 1px solid var(--line); border-radius: 1rem; padding: .05rem .6rem; font-size: .85rem; }
.heading { color: var(--muted); margin: 0 0 .5rem; }
ol.hits, ul.hits, ul.folders { list-style: none; margin: 0; padding: 0; }
.hits li { padding: .5rem 0; border-bottom: 1px solid var(--line); }
.hits h2 { font-size: 1rem; margin: 0; }
.hits p { margin: .15rem 0; }
.id { color: var(--muted); font-size: .8rem; word-break: break-all; }
.folders li { padding: .15rem 0; }
mark { background: var(--mark); color: inherit; }
.more { display: inline-block; margin: .8rem 0; }
.error { border-left: 3px solid #c33; padding: .3rem .8rem; }
.error .fix { color: var(--muted); }
.reader header { position: static; padding: 0; border: 0; }
.reader h1 { font-size: 1.3rem; margin: 0; }
.actions { display: flex; gap: 1rem; margin: .5rem 0; }
dl.fields { display: grid; grid-template-columns: max-content 1fr; gap: .1rem .8rem; font-size: .9rem; }
dl.fields div { display: contents; }
dl.fields dt { color: var(--muted); }
dl.fields dd { margin: 0; overflow-wrap: anywhere; }
dl.fields dd a + a::before { content: ", "; color: var(--muted); }
.text { font: 13px/1.5 ui-monospace, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
.text .n { display: inline-block; width: 3.5rem; color: var(--muted); user-select: none; }
.text .hit { background: #fd42; }
.text .h1, .text .h2, .text .h3 { font-weight: 700; }
.part { color: var(--muted); font-size: .85rem; }
@media (max-width: 60rem) { main { grid-template-columns: 1fr; } }
"""
