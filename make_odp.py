import zipfile, re

TEMPLATE = r'C:/Users/lcw/Desktop/AI_Code/CSP_YOLO/ES711模板.odp'
OUTPUT   = r'C:/Users/lcw/Desktop/AI_Code/CSP_YOLO/CSPPartial-YOLO_Report_v1.1.odp'
MD_FILE  = r'C:/Users/lcw/Desktop/AI_Code/CSP_YOLO/RESULTS_v1.1.md'

def esc(s):
    return (s.replace('&','&amp;').replace('<','&lt;')
             .replace('>','&gt;').replace('"','&quot;'))

def cover_slide(title, subtitle):
    return (
        '<draw:page draw:name="page_cover" draw:style-name="dp1" '
        'draw:master-page-name="封面" presentation:presentation-page-layout-name="AL1T1">'
        '<draw:frame presentation:style-name="pr1" draw:layer="layout" '
        'svg:width="30.482cm" svg:height="4cm" svg:x="1.846cm" svg:y="6cm" presentation:class="title">'
        '<draw:text-box><text:p text:style-name="P1"><text:span>' + esc(title) + '</text:span></text:p></draw:text-box>'
        '</draw:frame>'
        '<draw:frame presentation:style-name="pr2" draw:layer="layout" '
        'svg:width="30.482cm" svg:height="3.933cm" svg:x="1.846cm" svg:y="10.567cm" presentation:class="outline">'
        '<draw:text-box><text:p text:style-name="P1"><text:span>' + esc(subtitle) + '</text:span></text:p></draw:text-box>'
        '</draw:frame></draw:page>'
    )

def content_slide(page_id, title, lines):
    body = ''
    for line in lines:
        if line.strip() == '':
            body += '<text:p text:style-name="標題及內文-outline1"><text:span> </text:span></text:p>'
        elif line.startswith('    ') or line.startswith('\t'):
            body += ('<text:p text:style-name="標題及內文-outline2"><text:span>'
                     + esc(line.strip()) + '</text:span></text:p>')
        else:
            body += ('<text:p text:style-name="標題及內文-outline1"><text:span>'
                     + esc(line.strip()) + '</text:span></text:p>')
    return (
        '<draw:page draw:name="' + page_id + '" draw:style-name="dp1" '
        'draw:master-page-name="標題及內文" presentation:presentation-page-layout-name="AL1T1">'
        '<draw:frame presentation:style-name="pr4" draw:layer="layout" '
        'svg:width="30.482cm" svg:height="3.18cm" svg:x="1.693cm" svg:y="1.209cm" presentation:class="title">'
        '<draw:text-box><text:p text:style-name="P1"><text:span>' + esc(title) + '</text:span></text:p></draw:text-box>'
        '</draw:frame>'
        '<draw:frame presentation:style-name="pr5" draw:layer="layout" '
        'svg:width="30.482cm" svg:height="12.095cm" svg:x="1.693cm" svg:y="4.838cm" presentation:class="outline">'
        '<draw:text-box>' + body + '</draw:text-box>'
        '</draw:frame></draw:page>'
    )

# ── 解析 Markdown，每個 ## 段落 = 一張投影片 ──────────────
with open(MD_FILE, encoding='utf-8') as f:
    raw = f.read()

# 移除 code block（``` ... ```），保留說明文字
raw = re.sub(r'```.*?```', '', raw, flags=re.DOTALL)
# 移除 > 引用符號
raw = re.sub(r'^>\s*', '', raw, flags=re.MULTILINE)
# 移除 ** 粗體標記
raw = re.sub(r'\*\*(.+?)\*\*', r'\1', raw)
# 移除 ` 行內 code 標記
raw = re.sub(r'`([^`]+)`', r'\1', raw)
# 移除水平分隔線
raw = re.sub(r'^---+$', '', raw, flags=re.MULTILINE)
# 表格行保留，移除 |---|---| 分隔行
raw = re.sub(r'^\|[-| :]+\|$', '', raw, flags=re.MULTILINE)

lines_all = raw.split('\n')

# 切分成段落：H1 = 封面，H2 = 新投影片
slides = []
cur_title = ''
cur_lines = []
page_counter = [0]

def flush(title, lines):
    page_counter[0] += 1
    pid = 'p' + str(page_counter[0])
    # 過濾空行連續超過1行
    cleaned = []
    prev_empty = False
    for l in lines:
        is_empty = l.strip() == ''
        if is_empty and prev_empty:
            continue
        cleaned.append(l)
        prev_empty = is_empty
    slides.append(content_slide(pid, title, cleaned))

for line in lines_all:
    if line.startswith('# ') and not line.startswith('## '):
        # H1 = 封面
        title_text = line[2:].strip()
        slides.append(cover_slide(title_text,
            'IEEE JSTARS 2024  |  DOTA 4-class OBB Detection  |  PyTorch 復現'))
    elif line.startswith('## '):
        if cur_title:
            flush(cur_title, cur_lines)
        cur_title = line[3:].strip()
        cur_lines = []
    elif line.startswith('### '):
        # H3 當作內容行，加粗標題感
        cur_lines.append('[' + line[4:].strip() + ']')
    else:
        # 表格行：把 | 分隔的內容轉成縮排文字
        if line.strip().startswith('|') and line.strip().endswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            cells = [c for c in cells if c]
            if cells:
                cur_lines.append('  ' + '  |  '.join(cells))
        else:
            cur_lines.append(line)

if cur_title:
    flush(cur_title, cur_lines)

# ── 組裝並輸出 ODP ──────────────────────────────────────────
z_in = zipfile.ZipFile(TEMPLATE, 'r')
original = z_in.read('content.xml').decode('utf-8')

idx_open  = original.find('<office:presentation>')
prefix    = original[:idx_open + len('<office:presentation>')]
suffix    = '</office:presentation></office:body></office:document-content>'

new_content = (prefix + '\n' + '\n'.join(slides) +
               '\n<presentation:settings presentation:mouse-visible="false"/>\n' + suffix)

with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as z_out:
    for item in z_in.infolist():
        if item.filename == 'content.xml':
            z_out.writestr(item, new_content.encode('utf-8'))
        else:
            z_out.writestr(item, z_in.read(item.filename))

z_in.close()
print('Done: ' + OUTPUT)
print('Slides: ' + str(len(slides)))
