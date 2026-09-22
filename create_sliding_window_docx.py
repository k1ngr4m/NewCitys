from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from html import escape


out = Path(
    "D:/YimingLin/Jiamin.Zheng/专利/第二版专利/滑动窗口自适应调整说明.docx"
)
out.parent.mkdir(parents=True, exist_ok=True)

paras = [
    ("title", "滑动时间窗口自适应调整的具体实施方式"),
    (
        "normal",
        "在一实施例中，为使动态流量交互图能够同时兼顾平稳交通状态下的相关性估计稳定性和突发交通状态下的响应速度，所述滑动时间窗口的长度可根据节点交通序列的波动程度自适应调整。",
    ),
    (
        "normal",
        "设交通节点数为 N，第 i 个交通节点在时刻 t 的节点交通序列表示为 x_i(t)。在每个动态图更新时刻 t，先基于一个参考窗口计算节点交通序列的波动程度。具体地，可采用节点序列一阶差分的平均绝对变化量作为波动度量：",
    ),
    (
        "formula",
        r"$$v_i(t)=\frac{1}{T_0-1}\sum_{\tau=t-T_0+2}^{t}\left|x_i(\tau)-x_i(\tau-1)\right|$$",
    ),
    (
        "normal",
        "其中，T_0 表示用于计算波动程度的参考窗口长度，v_i(t) 表示第 i 个交通节点在当前时刻的波动程度。进一步地，对所有交通节点的波动程度求平均，得到当前路网整体波动程度：",
    ),
    ("formula", r"$$V(t)=\frac{1}{N}\sum_{i=1}^{N}v_i(t)$$"),
    (
        "normal",
        "根据整体波动程度 V(t) 与预设波动阈值 θ_v 的比较结果，自适应确定用于生成动态流量交互图的滑动时间窗口长度：",
    ),
    (
        "formula",
        r"$$T(t)=\begin{cases}T_{long}, & V(t)<\theta_v \\ T_{short}, & V(t)\geq\theta_v\end{cases}$$",
    ),
    (
        "normal",
        "其中，T_long 表示较长滑动时间窗口，T_short 表示较短滑动时间窗口，且 T_long > T_short。当交通序列波动程度较低时，采用较长滑动时间窗口，以提高节点相关性估计的稳定性；当交通序列波动程度较高时，采用较短滑动时间窗口，以增强动态流量交互图对突发变化的响应能力。",
    ),
    (
        "normal",
        "在确定当前滑动时间窗口长度 T(t) 后，截取各交通节点在该窗口内的节点交通序列，并计算任意两个交通节点之间的相关性。以皮尔逊相关系数为例，第 i 个节点和第 j 个节点之间的相关性得分可表示为：",
    ),
    (
        "formula",
        r"$$r_{ij}(t)=\frac{\sum_{\tau=t-T(t)+1}^{t}\left(x_i(\tau)-\bar{x}_i\right)\left(x_j(\tau)-\bar{x}_j\right)}{\sqrt{\sum_{\tau=t-T(t)+1}^{t}\left(x_i(\tau)-\bar{x}_i\right)^2}\sqrt{\sum_{\tau=t-T(t)+1}^{t}\left(x_j(\tau)-\bar{x}_j\right)^2}+\varepsilon}$$",
    ),
    (
        "normal",
        "其中，\\bar{x}_i 和 \\bar{x}_j 分别表示节点 i 和节点 j 在当前滑动时间窗口内的序列均值，ε 为防止分母为零的极小常数。",
    ),
    (
        "normal",
        "进一步地，根据相关性得分构建动态流量交互图的邻接矩阵。可对每个交通节点保留相关性得分最高的 K 个关联节点，或保留相关性得分高于预设阈值的节点对。以 Top-K 筛选为例：",
    ),
    (
        "formula",
        r"$$A_{flow}(i,j,t)=\begin{cases}\max(r_{ij}(t),0), & j\in TopK_i(t) \\ 0, & \text{otherwise}\end{cases}$$",
    ),
    (
        "normal",
        "其中，A_flow(i,j,t) 表示时刻 t 下动态流量交互图中节点 i 与节点 j 之间的边权，TopK_i(t) 表示与节点 i 相关性得分最高的 K 个节点集合。由此得到的动态流量交互图能够随交通状态变化自适应更新，在平稳状态下保持较稳定的关联结构，在波动较强或突发状态下提高对短期变化的敏感性。",
    ),
]


def p_xml(kind: str, text: str) -> str:
    text = escape(text)
    if kind == "title":
        return (
            '<w:p><w:pPr><w:pStyle w:val="Title"/><w:jc w:val="center"/></w:pPr>'
            '<w:r><w:rPr><w:b/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>'
            f"<w:t>{text}</w:t></w:r></w:p>"
        )
    if kind == "formula":
        return (
            '<w:p><w:pPr><w:spacing w:before="120" w:after="120"/>'
            '<w:jc w:val="center"/></w:pPr><w:r><w:rPr>'
            '<w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>'
            '<w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
            f'<w:t xml:space="preserve">{text}</w:t></w:r></w:p>'
        )
    return (
        '<w:p><w:pPr><w:spacing w:after="120"/><w:ind w:firstLine="420"/>'
        '</w:pPr><w:r><w:rPr><w:sz w:val="24"/><w:szCs w:val="24"/>'
        f"</w:rPr><w:t>{text}</w:t></w:r></w:p>"
    )


body = "\n".join(p_xml(k, t) for k, t in paras)

document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:w10="urn:schemas-microsoft-com:office:word" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" mc:Ignorable="w14 wp14"><w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body></w:document>'''

content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''

rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''

doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'''

styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:eastAsia="宋体" w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style></w:styles>'''

with ZipFile(out, "w", ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", content_types.encode("utf-8"))
    z.writestr("_rels/.rels", rels.encode("utf-8"))
    z.writestr("word/_rels/document.xml.rels", doc_rels.encode("utf-8"))
    z.writestr("word/document.xml", document_xml.encode("utf-8"))
    z.writestr("word/styles.xml", styles.encode("utf-8"))

print(out)
