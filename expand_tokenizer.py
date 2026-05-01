# expand_tokenizer.py
# 用途：在 tokenizer/ 分词器基础上扩充词表
# 新增优先级：LaTeX 符号（主）> 英语词根词缀 + 科技词汇（次）> 化学符号（辅）
# 目标词表大小：53376（= 128 × 417，对齐 tensor 计算）
# 输出：tokenizer/ 目录下的 tokenizer.json 和 tokenizer_config.json

import json                          # 读写 tokenizer.json（BPE 词表 + added_tokens）
import math                          # 数学运算（当前版本未直接使用，预留）
import shutil                        # 复制 tokenizer_config.json 到输出目录
from pathlib import Path             # 面向对象的路径操作

# ─── 路径配置 ────────────────────────────────────────────────────────
源分词器路径 = Path(__file__).parent / "tokenizer" / "tokenizer.json"  # 上游分词器文件（52013 tokens）
源配置路径   = Path(__file__).parent / "tokenizer" / "tokenizer_config.json"  # 上游分词器配置
输出目录     = Path(__file__).parent / "tokenizer"  # 扩充后的分词器输出目录（与源目录相同，写回覆盖）

目标词表大小 = 53376    # = 128 × 420

# ─── 新增 token 定义（优先级从高到低）───────────────────────────────

LATEX_TOKENS = [
    # ── 希腊字母小写 ──
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon",
    r"\zeta", r"\eta", r"\theta", r"\vartheta", r"\iota", r"\kappa",
    r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\varpi",
    r"\rho", r"\varrho", r"\sigma", r"\varsigma", r"\tau", r"\upsilon",
    r"\phi", r"\varphi", r"\chi", r"\psi", r"\omega",
    # ── 希腊字母大写 ──
    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi", r"\Pi",
    r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi", r"\Omega",
    # ── 数学运算符 ──
    r"\frac", r"\sqrt", r"\int", r"\sum", r"\prod", r"\partial",
    r"\nabla", r"\infty", r"\pm", r"\mp", r"\times", r"\div",
    r"\cdot", r"\circ", r"\oplus", r"\otimes", r"\wedge", r"\vee",
    # ── 关系符号 ──
    r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv", r"\cong",
    r"\sim", r"\simeq", r"\propto", r"\subset", r"\supset",
    r"\subseteq", r"\supseteq", r"\in", r"\notin", r"\ni",
    r"\prec", r"\succ", r"\preceq", r"\succeq", r"\ll", r"\gg",
    r"\parallel", r"\perp", r"\models", r"\vdash", r"\dashv",
    # ── 逻辑与箭头 ──
    r"\forall", r"\exists", r"\nexists", r"\neg", r"\land", r"\lor",
    r"\Rightarrow", r"\Leftarrow", r"\Leftrightarrow",
    r"\rightarrow", r"\leftarrow", r"\leftrightarrow",
    r"\mapsto", r"\to", r"\gets",
    r"\uparrow", r"\downarrow", r"\Uparrow", r"\Downarrow",
    r"\nearrow", r"\searrow", r"\nwarrow", r"\swarrow",
    r"\hookrightarrow", r"\hookleftarrow",
    r"\rightharpoonup", r"\rightharpoondown",
    r"\leftharpoonup", r"\leftharpoondown",
    r"\rightleftharpoons", r"\iff",
    # ── 集合与数域 ──
    r"\mathbb{R}", r"\mathbb{Z}", r"\mathbb{N}", r"\mathbb{Q}",
    r"\mathbb{C}", r"\mathbb{F}", r"\mathbb{P}", r"\mathbb{E}",
    r"\emptyset", r"\varnothing", r"\cup", r"\cap",
    r"\setminus", r"\complement", r"\sqcup", r"\sqcap",
    r"\bigcup", r"\bigcap", r"\bigsqcup",
    # ── 数学函数 ──
    r"\lim", r"\max", r"\min", r"\sup", r"\inf", r"\arg",
    r"\det", r"\exp", r"\log", r"\ln", r"\lg",
    r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc",
    r"\arcsin", r"\arccos", r"\arctan",
    r"\sinh", r"\cosh", r"\tanh", r"\coth",
    r"\gcd", r"\lcm", r"\deg",
    r"\limsup", r"\liminf", r"\lim_{}", r"\argmax", r"\argmin",
    r"\ker", r"\hom", r"\dim", r"\rank", r"\tr", r"\Pr",
    r"\Re", r"\Im", r"\operatorname",
    # ── 格式命令 ──
    r"\text", r"\mathrm", r"\mathbf", r"\mathit",
    r"\mathcal", r"\mathscr", r"\mathfrak", r"\mathbb",
    r"\hat", r"\bar", r"\tilde", r"\vec",
    r"\overline", r"\underline", r"\widehat", r"\widetilde",
    r"\dot", r"\ddot", r"\acute", r"\grave", r"\check",
    r"\breve", r"\mathring",
    r"\textbf", r"\textit", r"\textrm", r"\texttt",
    r"\emph", r"\textsf",
    r"\color", r"\textcolor", r"\colorbox",
    r"\boxed", r"\displaystyle", r"\textstyle",
    # ── 括号 ──
    r"\left(", r"\right)", r"\left[", r"\right]",
    r"\left\{", r"\right\}", r"\langle", r"\rangle",
    r"\lvert", r"\rvert", r"\lVert", r"\rVert",
    r"\lceil", r"\rceil", r"\lfloor", r"\rfloor",
    r"\bigl", r"\bigr", r"\Bigl", r"\Bigr",
    r"\biggl", r"\biggr", r"\Biggl", r"\Biggr",
    r"\left.", r"\right.",
    # ── 环境标记 ──
    r"\begin{equation}", r"\end{equation}",
    r"\begin{align}", r"\end{align}",
    r"\begin{aligned}", r"\end{aligned}",
    r"\begin{matrix}", r"\end{matrix}",
    r"\begin{pmatrix}", r"\end{pmatrix}",
    r"\begin{bmatrix}", r"\end{bmatrix}",
    r"\begin{vmatrix}", r"\end{vmatrix}",
    r"\begin{Vmatrix}", r"\end{Vmatrix}",
    r"\begin{cases}", r"\end{cases}",
    r"\begin{gather}", r"\end{gather}",
    r"\begin{gathered}", r"\end{gathered}",
    r"\begin{split}", r"\end{split}",
    r"\begin{array}", r"\end{array}",
    r"\begin{itemize}", r"\end{itemize}",
    r"\begin{enumerate}", r"\end{enumerate}",
    r"\begin{document}", r"\end{document}",
    r"\begin{figure}", r"\end{figure}",
    r"\begin{table}", r"\end{table}",
    # ── 常用数学记号 ──
    r"\binom", r"\pmod", r"\mod",
    r"\because", r"\therefore", r"\qed", r"\square",
    r"\cdots", r"\ldots", r"\vdots", r"\ddots",
    r"\quad", r"\qquad", r"\,", r"\;", r"\:",
    r"\label", r"\ref", r"\eqref", r"\tag",
    r"$$", r"\[", r"\]", r"\(", r"\)",
    r"\star", r"\ast", r"\bullet", r"\diamond",
    r"\dagger", r"\ddagger", r"\S", r"\P",
    r"\angle", r"\triangle", r"\Box",
    r"\hbar", r"\ell", r"\wp", r"\aleph",
    # ── 微积分 ──
    r"\frac{d}{dx}", r"\frac{\partial}{\partial}",
    r"\int_{}^{}", r"\sum_{}^{}", r"\prod_{}^{}",
    r"\oint", r"\iint", r"\iiint", r"\oiint",
    r"\lim_{n\to\infty}", r"\lim_{x\to}",
    r"\frac{\partial^2}{\partial x^2}",
    r"\nabla \times", r"\nabla \cdot", r"\nabla^2",
    # ── 线性代数 ──
    r"\mathbf{A}", r"\mathbf{x}", r"\mathbf{b}",
    r"\mathbf{I}", r"\mathbf{0}", r"\top",
    r"\otimes", r"\oplus", r"\odot",
    r"\text{tr}", r"\text{rank}", r"\text{diag}",
    r"\text{span}", r"\text{ker}", r"\text{im}", r"\text{null}",
    # ── 概率统计 ──
    r"\mathbb{E}", r"\mathbb{P}", r"\text{Var}", r"\text{Cov}",
    r"\overset{iid}{\sim}", r"\overset{d}{=}",
    r"\mathcal{N}", r"\mathcal{L}", r"\mathcal{O}",
    r"\mathcal{U}", r"\mathcal{B}", r"\mathcal{P}",
    r"\text{Pr}", r"\text{Bernoulli}", r"\text{Poisson}",
    r"\text{Binomial}", r"\text{Exponential}", r"\text{Uniform}",
    # ── 物理常用 ──
    r"\vec{F}", r"\vec{v}", r"\vec{a}", r"\vec{E}", r"\vec{B}",
    r"\vec{r}", r"\vec{p}", r"\vec{J}",
    r"\hbar", r"\hslash",
    r"\langle \psi |", r"| \phi \rangle",
    r"\bra{}", r"\ket{}", r"\braket{}",
    r"\hat{H}", r"\hat{p}", r"\hat{x}", r"\hat{L}", r"\hat{S}",
    r"\text{eV}", r"\text{MeV}", r"\text{GeV}",
    r"\text{Hz}", r"\text{kHz}", r"\text{MHz}", r"\text{GHz}",
    r"\text{kg}", r"\text{g}", r"\text{mol}", r"\text{K}",
    r"\text{Pa}", r"\text{J}", r"\text{W}", r"\text{V}", r"\text{A}",
    r"\text{m}", r"\text{cm}", r"\text{mm}", r"\text{nm}",
    r"\text{s}", r"\text{ms}", r"\text{ns}",
    # ── 计算机/算法常用 ──
    r"\mathcal{O}", r"\mathcal{\Omega}", r"\mathcal{\Theta}",
    r"\text{iff}", r"\text{s.t.}", r"\text{w.r.t.}", r"\text{i.e.}",
    r"\text{e.g.}", r"\text{cf.}", r"\text{etc.}",
    r"\text{LHS}", r"\text{RHS}",
]

ENGLISH_TOKENS = [
    # ═══════════════════════════════════════════════════════════════
    # 第一部分：基础科技词汇
    # ═══════════════════════════════════════════════════════════════
    # 数学
    "theorem", "proof", "lemma", "corollary", "definition",
    "proposition", "hypothesis", "conjecture", "axiom", "remark",
    "matrix", "vector", "tensor", "scalar", "eigenvalue",
    "eigenvector", "determinant", "transpose", "inverse",
    "derivative", "integral", "gradient", "divergence", "curl",
    "differential", "polynomial", "sequence", "series",
    "convergence", "divergence", "continuity", "limit",
    "probability", "distribution", "variance", "deviation",
    "expectation", "covariance", "correlation", "entropy",
    "dataset", "training", "validation", "optimization",
    "function", "parameter", "variable", "constant",
    "algorithm", "complexity", "approximation",
    "maximum", "minimum", "optimal", "constraint",
    "permutation", "combination", "bijection", "surjection",
    "isomorphism", "homomorphism", "morphology", "topology",
    "manifold", "metric", "norm", "metric",
    "convex", "concave", "monotonic", "bounded",
    "symmetric", "asymmetric", "orthogonal", "commutative",
    "associative", "distributive", "idempotent",
    "recurrence", "induction", "contradiction", "necessity",
    "sufficiency", "equivalence", "implication",
    # 物理
    "molecule", "electron", "proton", "neutron",
    "reaction", "catalyst", "equilibrium",
    "wavelength", "frequency", "amplitude", "momentum",
    "acceleration", "velocity", "displacement", "torque",
    "thermodynamics", "kinematics", "dynamics", "mechanics",
    "electromagnetic", "photoelectric", "interference",
    "diffraction", "refraction", "reflection", "polarization",
    "relativity", "quantum", "chromatic", "doppler",
    "semiconductor", "superconductor", "insulator", "conductor",
    # 生物
    "mitosis", "meiosis", "chromosome", "genome",
    "metabolism", "photosynthesis", "respiration",
    "homeostasis", "evolution", "speciation",
    "protein", "enzyme", "amino", "nucleotide",
    "ribosome", "mitochondria", "chloroplast",
    "neuron", "synapse", "hormone", "receptor",
    "antibody", "antigen", "pathogen", "vaccine",
    "ecosystem", "biodiversity", "biosphere",
    # 计算机
    "recursion", "iteration", "heuristic", "greedy",
    "backpropagation", "activation", "normalization",
    "regularization", "augmentation", "initialization",
    "architecture", "encoder", "decoder", "embedding",
    "attention", "transformer", "convolution", "pooling",
    "sigmoid", "softmax", "relu", "gelu", "tanh",
    "overfitting", "underfitting", "generalization",
    "hyperparameter", "epoch", "batch", "learning",
    "inference", "deployment", "benchmark", "ablation",

    # ═══════════════════════════════════════════════════════════════
    # 第二部分：词根（Root）—— 英语构词的核心部件
    # ═══════════════════════════════════════════════════════════════
    # 拉丁词根
    "act",       # 驱动（activate, react, interact）
    "agri",      # 农业（agriculture, agrarian）
    "alt",       # 高（altitude, altimeter）
    "am",        # 爱（amiable, amicable）
    "anim",      # 生命/精神（animate, unanimous）
    "ann",       # 年（annual, anniversary, perennial）
    "aqu",       # 水（aquarium, aquatic, aqueduct）
    "aud",       # 听（audible, audience, auditorium）
    "bell",      # 战争（belligerent, rebellion, antebellum）
    "brev",      # 短（abbreviate, brevity）
    "cad",       # 落下（cadence, cascade, decadent）
    "cap",       # 拿/取（capture, capable, capacity）
    "card",      # 心（cardiac, cardinal, discard）
    "ced",       # 走/让出（precede, recede, concede）
    "cent",      # 百（century, percent, centigrade）
    "cern",      # 分离/辨别（concern, discern, discriminate）
    "cert",      # 确定（certain, certify, certificate）
    "chron",     # 时间（chronicle, synchronize, chronic）
    "cid",       # 切/落（decide, incident, coincidence）
    "circ",      # 环（circle, circuit, circumference）
    "claim",     # 喊（exclaim, proclaim, acclaim")
    "clin",      # 倾斜（decline, incline, recline）
    "cogn",      # 知道（recognize, cognition, agnostic）
    "cord",      # 心（accord, concord, discord）
    "corp",      # 身体（corporation, corpse, corporeal）
    "cred",      # 相信（credit, credible, incredible）
    "cruc",      # 十字（crucial, crucify, cruciform）
    "cub",       # 躺/孵（incubate, succumb, concubine）
    "culp",      # 罪（culprit, culpable, exculpate）
    "cur",       # 跑（current, occur, recur, cursor）
    "dic",       # 说（dictionary, predict, indicate）
    "doc",       # 教（document, doctrine, docile）
    "duc",       # 引导（produce, reduce, conduct, educate）
    "dur",       # 持久（durable, duration, endure, obdurate）
    "equ",       # 平等（equal, equity, equivalent, equation）
    "err",       # 错误（error, errant, aberrant）
    "fac",       # 做（factory, manufacture, facile）
    "fall",      # 欺骗（fallacy, fallible, infallible）
    "feder",     # 联盟（federal, federation, confederate）
    "fer",       # 带来（transfer, refer, confer, differ）
    "fid",       # 信任（fidelity, confidence, fidelity）
    "fin",       # 结束（final, finish, infinite, define）
    "flect",     # 弯曲（reflect, deflect, inflect）
    "form",      # 形状（reform, conform, uniform, formula）
    "fort",      # 强（fortify, comfort, effort, fortress）
    "fract",     # 破碎（fraction, fracture, fragment, infract）
    "fug",       # 逃（refuge, fugitive, centrifuge）
    "fund",      # 底部（foundation, fundamental, profound）
    "gen",       # 产生/种类（generate, genesis, genetic, generic）
    "grad",      # 步/级（grade, gradual, graduate, degrade）
    "graph",     # 写/画（graph, graphic, photograph, telegraph）
    "grat",      # 高兴/恩惠（grateful, gratify, gratitude, ingratiate）
    "grav",      # 重（gravity, grave, aggravate）
    "greg",      # 群（aggregate, congregate, segregate, egregious）
    "gress",     # 走（progress, congress, digress, transgress）
    "hab",       # 持有/居住（habit, inhabit, habitat, rehabilitate")
    "hibit",     # 持有（exhibit, prohibit, inhibit, prohibit）
    "hydr",      # 水（hydrate, hydrogen, hydrology, dehydrate")
    "ject",      # 投（project, inject, reject, subject, deject）
    "jud",       # 判断（judge, judicial, prejudice, adjudicate）
    "junct",     # 连接（junction, conjunction, adjunct, disjunct）
    "jur",       # 法律（jury, jurisdiction, perjure, abjure）
    "labor",     # 工作（labor, laboratory, collaborate, elaborate）
    "lat",       # 携带（relate, translate, collate, dilate）
    "lect",      # 选/读（elect, collect, select, dialect）
    "leg",       # 法律（legal, legislation, legitimate, allege）
    "lev",       # 轻/升（elevate, alleviate, lever, levee）
    "liber",     # 自由（liberty, liberal, liberate, deliberate）
    "lingu",     # 语言（linguistic, bilingual, language）
    "liter",     # 文字（literary, literal, literature, illiterate）
    "loc",       # 地方（location, allocate, locality, relocate）
    "log",       # 言/学（logic, dialogue, prologue, catalogue）
    "luc",       # 光（lucent, elucidate, translucent, lucid）
    "magn",      # 大（magnify, magnificent, magnitude）
    "man",       # 手（manual, manufacture, manuscript, manacle）
    "mand",      # 命令（command, mandate, demand, remand）
    "mari",      # 海（marine, submarine, mariner, maritime）
    "matr",      # 母（matrix, maternal, matriarch, maternity）
    "medi",      # 中间（mediate, medium, medieval, immediate）
    "mem",       # 记忆（memory, memoir, memorable, commemorate）
    "mend",      # 错误/修复（amend, mendacious, emend）
    "ment",      # 心智（mental, mentality, mentor, demented）
    "merc",      # 商业（merchant, commerce, mercenary, mercurial）
    "merg",      # 沉没（merge, submerge, emerge, immerse）
    "migr",      # 移动（migrate, immigrant, emigrate, transmigrate）
    "min",       # 小（minor, diminish, minute, diminish）
    "misc",      # 混杂（miscellaneous, miscible, promiscuous）
    "miss",      # 送（mission, submit, transmit, dismiss, admit）
    "mit",       # 送（commit, permit, submit, admit, remit）
    "mob",       # 移动（mobile, automobile, mob, mobilize）
    "mon",       # 提醒/告诫（monitor, monument, admonish, premonition）
    "mort",      # 死（mortal, mortify, mortgage, mortuary）
    "mov",       # 移动（move, remove, promote, remote）
    "mut",       # 改变（mutate, mutual, commute, immutable）
    "nat",       # 出生（native, nature, innate, prenatal）
    "nav",       # 船（navigate, navy, naval, circumnavigate）
    "neg",       # 否定（negate, negative, renegate, abnegate）
    "nomin",     # 名字（nominate, nomenclature, nominal, ignominious）
    "norm",      # 规范（normal, enormous, enormous, abnormal）
    "nov",       # 新（novel, novice, innovate, renovate）
    "numer",     # 数（number, numerical, enumerate, innumerable）
    "nunci",     # 宣告（announce, pronounce, enunciate, denunciate）
    "oper",      # 工作（operate, cooperate, opera, inoperable）
    "opt",       # 最好/眼（optimal, optics, optician, optimistic）
    "ord",       # 顺序（order, ordinary, coordinate, subordinate）
    "ori",       # 起源（origin, orient, orientation, disorient）
    "pac",       # 和平（pacific, pacify, pace, pacifist）
    "par",       # 准备/出现（prepare, compare, apparent, separate）
    "part",      # 部分（part, partial, partition, counterpart）
    "pass",      # 经过（passage, surpass, compass, bypass）
    "patr",      # 父（paternal, patriarch, patron, patriot）
    "ped",       # 脚（pedal, pedestrian, pedestal, expedite）
    "pel",       # 推（repel, compel, expel, dispel, propel）
    "pend",      # 悬挂（depend, suspend, append, pending, pension）
    "petr",      # 石（petroleum, petrify, petrochemical）
    "phil",      # 爱（philosophy, philanthropy, bibliophile）
    "pict",      # 画（picture, depict, picturesque, pictograph）
    "plac",      # 取悦/平静（placebo, placate, placid, implacable）
    "plic",      # 折叠（complicate, replicate, explicit, implicit, apply）
    "pon",       # 放（component, opponent, postpone, exponent）
    "popul",     # 人民（population, popular, populace, populate）
    "port",      # 携带（transport, import, export, deport, report）
    "pos",       # 放（position, compose, deposit, dispose, propose）
    "pot",       # 能力（potential, potent, impotent, omnipotent）
    "prehen",    # 抓（comprehend, apprehend, reprehend）
    "prim",      # 第一（primary, prime, primitive, primer）
    "priv",      # 私人（private, privilege, deprive, privy）
    "prob",      # 试/证明（problem, probable, probe, approbate）
    "proper",    # 自己的/适当（property, appropriate, proper, propriety）
    "pugn",      # 打（pugnacious, repugnant, impugn, oppugn）
    "punct",     # 点（punctual, punctuate, punctuation, acupuncture）
    "pur",       # 纯净（pure, purify, purge, impure, purport）
    "put",       # 思考/修剪（compute, dispute, reputation, amputate）
    "quer",      # 询问（query, quest, inquire, require）
    "qui",       # 安静（quiet, tranquil, acquiesce, requiem）
    "rupt",      # 破裂（rupture, interrupt, corrupt, erupt, abrupt）
    "sacr",      # 神圣（sacred, sacrifice, sacrament, desecrate）
    "sal",       # 盐/跳（salary, saline, salient, resilient）
    "sanct",     # 神圣（sanction, sanctuary, sanctify, sanctum")
    "sat",       # 满足（satisfy, saturate, sated, insatiable）
    "scend",     # 爬（ascend, descend, transcend, condescend）
    "sci",       # 知道（science, conscious, omniscient, prescient）
    "scrib",     # 写（describe, prescribe, subscribe, inscribe, scribe）
    "sed",       # 坐（sediment, sedate, supersede, reside）
    "sens",      # 感觉（sense, sensitive, sensory, consent, dissent）
    "sequ",      # 跟随（sequence, consequence, subsequent, sequel）
    "serv",      # 服务/保存（serve, preserve, conserve, observe, reserve）
    "sign",      # 标记（signal, signature, significant, designate）
    "simil",     # 相似（similar, simulate, assimilate, simile）
    "sol",       # 单独/太阳（solar, solitude, solitary, soliloquy）
    "solv",      # 松开（solve, dissolve, resolve, solvent, absolve）
    "soph",      # 智慧（philosophy, sophisticated, sophomore）
    "spec",      # 看（inspect, spectator, spectrum, spectacular, specious）
    "spir",      # 呼吸（spirit, inspire, respire, conspire, aspire）
    "sta",       # 站立（stable, station, status, substance, constant）
    "struct",    # 建造（structure, construct, instruct, destruct, obstruct）
    "sum",       # 拿（assume, consume, resume, presume, sumptuous）
    "tact",      # 触（contact, tactile, intact, contingent）
    "tain",      # 持有（contain, obtain, maintain, sustain, retain）
    "techn",     # 技术（technique, technology, technical, polytechnic）
    "tempor",    # 时间（temporary, contemporary, temporal, extempore）
    "ten",       # 持有（tenant, tenacious, tenure, content, detention）
    "tend",      # 伸展（tend, extend, contend, intend, tendency）
    "terr",      # 土地/恐惧（territory, terrain, terrestrial, terrible）
    "test",      # 证据/测试（testify, attest, contest, testament）
    "text",      # 编织（text, textile, context, pretext, texture）
    "theo",      # 神（theology, theocracy, atheism, apotheosis）
    "therm",     # 热（thermal, thermometer, thermostat, hypothermia）
    "torq",      # 扭转（torque, torsion, distort, contort, extort）
    "tort",      # 扭（torture, tortuous, distort, contort, retort）
    "tour",      # 转（tour, tourism, contour, detour）
    "tox",       # 毒（toxic, toxicology, detoxify, intoxicate）
    "tract",     # 拉（attract, extract, subtract, contract, tractor）
    "trib",      # 给予（contribute, distribute, attribute, tributary）
    "trud",      # 推（intrude, extrude, protrude, obtrude）
    "turb",      # 搅乱（turbine, turbulent, disturb, perturb）
    "umbr",      # 影子（umbrella, umbrage, adumbrate, penumbra）
    "und",       # 浪（abound, redundant, undulate, inundate）
    "uni",       # 一（unite, uniform, unique, universal, unicorn）
    "urb",       # 城市（urban, suburb, urbanize, urbane）
    "util",      # 用（utility, utilize, utilization, utilitarian）
    "vac",       # 空（vacant, vacuum, evacuate, vacuous, vacation）
    "vad",       # 走（invade, evade, pervade）
    "val",       # 强/价值（value, valid, evaluate, equivalent, prevalent）
    "ven",       # 来（advent, convention, revenue, intervene, venue）
    "ver",       # 真实（verify, verity, veracious, aver, verdict）
    "verb",      # 词（verbal, verbose, proverb, verbatim）
    "vers",      # 转（reverse, convert, diverse, universe, version）
    "vert",      # 转（convert, revert, invert, subvert, avert）
    "vest",      # 衣服（vest, vestment, invest, divest）
    "vi",        # 路（via, deviate, previous, trivial, viable）
    "vid",       # 看（video, evidence, provide, evident, provident）
    "vinc",      # 征服（convince, evince, invincible, province）
    "vis",       # 看（vision, visible, revise, supervise, visual）
    "vit",       # 生命（vital, vitamin, vitality, vitalize）
    "viv",       # 活（vivid, revive, survive, vivacious, vivify）
    "voc",       # 声音/叫（vocal, vocabulary, advocate, evoke, provoke）
    "vol",       # 意愿（volition, benevolent, malevolent, voluntary）
    "volv",      # 转（revolve, evolve, involve, devolve, convolve）

    # 希腊词根
    "acr",       # 尖/高（acrid, acrobat, acrimony, acronym）
    "aer",       # 空气（aerial, aerate, aerobic, aerosol）
    "anthrop",   # 人（anthropology, misanthrope, philanthropy）
    "arch",      # 统治/首要（monarch, hierarchy, archetype, archive）
    "aster",     # 星（asterisk, astronomy, asteroid, disaster）
    "auto",      # 自己（automatic, autonomy, automobile, autobiography）
    "bio",       # 生命（biology, biography, biochemistry, symbiosis）
    "cardi",     # 心（cardiac, cardiology, electrocardiogram）
    "cephal",    # 头（encephalitis, cephalopod, microcephaly）
    "chrom",     # 颜色（chromosome, monochrome, chromatic, polychrome）
    "chron",     # 时间（chronic, chronicle, synchronize, anachronism）
    "cosm",      # 宇宙（cosmic, cosmology, microcosm, cosmopolitan）
    "crat",      # 统治（democrat, aristocrat, autocrat, bureaucrat）
    "crypt",     # 隐藏（cryptic, encrypt, crypt, apocryphal）
    "cycl",      # 环/圆（cycle, bicycle, cyclone, encyclical）
    "dem",       # 人民（democracy, demographic, epidemic, endemic）
    "derm",      # 皮（dermatology, epidermis, hypodermic, pachyderm）
    "dynam",     # 力（dynamic, dynamite, dynamo, hydrodynamics）
    "eco",       # 家/环境（ecology, economy, ecosystem, ecotourism）
    "esthes",    # 感觉（aesthetic, anesthesia, kinesthesia, synesthesia）
    "gam",       # 婚姻（monogamy, polygamy, bigamy, cryptogam）
    "gastr",     # 胃（gastric, gastronomy, gastropod, gastritis）
    "gen",       # 产生/种族（genesis, gene, generate, pathogen, gender）
    "geo",       # 地球（geology, geography, geocentric, geometry）
    "gnos",      # 知（diagnosis, prognosis, agnostic, gnostic）
    "gon",       # 角（polygon, pentagon, trigonometry, diagonal）
    "gram",      # 写/画（telegram, diagram, grammar, program）
    "graph",     # 写/记录（graphic, photograph, autograph, graphite）
    "gyn",       # 女（gynecology, androgyny, misogyny）
    "hal",       # 呼吸（inhale, exhale, halitosis）
    "hedon",     # 快乐（hedonism, hedonist）
    "helio",     # 太阳（helium, heliocentric, heliotrope）
    "hemo",      # 血（hemoglobin, hemorrhage, hemophilia, anemia）
    "hepat",     # 肝（hepatitis, hepatology, hepatocyte）
    "hetero",    # 异（heterogeneous, heterodox, heterosexual）
    "hist",      # 组织（histology, histogram, histamine）
    "hom/o",     # 同（homogeneous, homonym, homosexual, homeostasis）
    "hydr",      # 水（hydrogen, hydrology, dehydrate, hydrant）
    "hyper",     # 超/上（hyperactive, hypertension, hyperlink）
    "hypno",     # 睡（hypnosis, hypnotic, hypnotize）
    "icon",      # 像（icon, iconic, iconoclast, iconography）
    "idio",      # 自己/独特（idiom, idiot, idiopathic, idiosyncrasy）
    "iso",       # 等（isotope, isometric, isomorphic, isolate）
    "kin",       # 运动（kinetic, kinematics, kinesthesia）
    "lith",      # 石（lithology, monolith, neolithic, aerolith）
    "log",       # 学/言（biology, prologue, catalogue, monologue）
    "lysis",     # 松开/分解（analysis, paralysis, electrolysis, catalysis）
    "matri",     # 母（matriarch, maternal, matrix, matrimony）
    "mega",      # 大（megabyte, megalopolis, megaphone, megaton）
    "melan",     # 黑（melanin, melancholy, melanoma, melancholic）
    "meter",     # 测量（thermometer, diameter, barometer, perimeter）
    "micro",     # 小（microscope, microbe, microorganism, microchip）
    "mime",      # 模仿（mime, mimic, mimeograph, pantomime）
    "mnemo",     # 记忆（mnemonic, amnesia）
    "mono",      # 一（monopoly, monotone, monologue, monolith）
    "morph",     # 形态（morphology, amorphous, metamorphosis, polymorphic）
    "multi",     # 多（multiply, multimedia, multitask, multilingual）
    "myo",       # 肌肉（myopia, myocardial, myoglobin）
    "narc",      # 麻木（narcotic, narcosis, narcolepsy）
    "neo",       # 新（neonatal, neoclassical, neologism, neophyte）
    "neur",      # 神经（neurology, neural, neuron, neurosis）
    "nom",       # 法则/管理（astronomy, economy, taxonomy, autonomy）
    "nym",       # 名（synonym, antonym, acronym, pseudonym, homonym）
    "oct",       # 八（octagon, octopus, octave, October）
    "odont",     # 牙（orthodontist, periodontal, odontology）
    "onom",      # 名（onomatopoeia, patronymic, metronymic）
    "ophthalm",  # 眼（ophthalmology, ophthalmic, ophthalmologist）
    "ornith",    # 鸟（ornithology, ornithologist）
    "oste",      # 骨（osteoporosis, osteopathy, osteocyte）
    "pach",      # 厚（pachyderm, pachymeningitis）
    "paleo",     # 古（paleolithic, paleontology, paleoecology）
    "pan",       # 全（panorama, pandemic, panacea, pantheon）
    "pap",       # 爸爸（papal, papacy, pope）
    "par/a",     # 旁边/保护（parallel, parasite, paramount, parachute）
    "path",      # 感受/疾病（pathology, empathy, apathy, psychopath）
    "patri",     # 父（patriot, patriarch, patron, compatriot）
    "ped/o",     # 儿童（pediatric, pedagogy, orthopedic）
    "pent",      # 五（pentagon, pentameter, pentathlon, Pentecost）
    "petr",      # 石（petroleum, petrify, petrochemical, petroglyph）
    "phag",      # 吃（phagocyte, esophagus, bacteriophage, anthropophagy）
    "phan",      # 显现（phantom, phantasm, diaphanous, phantom）
    "phil",      # 爱（philosophy, bibliophile, philanthropy, Philadelphia）
    "phob",      # 恐惧（phobia, claustrophobia, xenophobia, acrophobia）
    "phon",      # 声音（phonetic, telephone, symphony, cacophony）
    "phot",      # 光（photograph, photon, photosynthesis, photocopy）
    "phys",      # 自然/身体（physics, physiology, physical, physician）
    "plas",      # 塑造（plastic, plasma, plaster, neoplasm, metaplasia）
    "pneum",     # 肺/气（pneumonia, pneumatic, pneumothorax）
    "pod",       # 脚（podium, tripod, arthropod, podiatry）
    "polis",     # 城市（police, policy, politics, metropolitan, cosmopolis）
    "poly",      # 多（polygon, polyglot, polytechnic, polymer, polynomial）
    "proto",     # 第一（prototype, protocol, protagonist, protozoa）
    "pseud",     # 假（pseudonym, pseudoscience, pseudo）
    "psych",     # 心灵（psychology, psychic, psychotherapy, psychiatry）
    "pter",      # 翅/飞（pterodactyl, helicopter, opteryx）
    "pyr",       # 火（pyramid, pyrotechnics, antipyretic, pyromaniac）
    "sarc",      # 肉（sarcasm, sarcophagus, sarcoma）
    "scop",      # 看（scope, microscope, telescope, kaleidoscope）
    "seism",     # 地震（seismology, seismic, seismograph）
    "semi",      # 半（semicircle, semiconductor, semifinal, semicolon）
    "somn",      # 睡（insomnia, somnolent, somnambulism）
    "soph",      # 智慧（philosophy, sophisticated, sophomore, sophistry）
    "spec",      # 看（spectacle, spectrum, spectator, introspection）
    "stereo",    # 立体（stereophonic, stereotype, stereoscope）
    "strat",     # 层（strategy, strata, stratify, stratagem, prostrate）
    "syn",       # 共同（synthesis, synonym, synchronous, syndrome, synergy）
    "tact",      # 触（contact, tactile, intact, tactual）
    "tele",      # 远（telephone, telescope, television, telepathy）
    "therm",     # 热（thermal, thermometer, geothermal, exothermic）
    "tom",       # 切（atom, anatomy, appendectomy, epitome, dichotomy）
    "ton",       # 音调（tone, monotone, baritone, atonal, tonic）
    "top",       # 地方（topic, topology, utopia, topography, isotope）
    "tox",       # 毒（toxic, toxin, detoxify, toxicology, intoxicate")
    "typ",       # 型（type, typical, prototype, typography, atypical）
    "xen",       # 异/外（xenon, xenophobia, xenophile）
    "xer",       # 干（xerophyte, xerography, xeric）
    "xylo",      # 木（xylophone, xylem, xyloid）
    "zoo",       # 动物（zoology, zoo, zooplankton, protozoa）

    # ═══════════════════════════════════════════════════════════════
    # 第三部分：前缀（Prefix）
    # ═══════════════════════════════════════════════════════════════
    "a", "ab", "abs",       # 离开（avert, abnormal, abstract）
    "ad",                    # 向（advance, adjacent, adequate）
    "ambi",                  # 两者（ambiguous, ambidextrous, ambivalent）
    "amphi",                 # 周围（amphibian, amphitheater）
    "an", "ana",             # 无/向上（anemia, anarchy, anatomy, analogy）
    "ante",                  # 前（anterior, antecedent, antedate）
    "anti",                  # 反对（antibiotic, antibody, antisocial）
    "apo",                   # 离开（apology, apostle, apotheosis, apocryphal）
    "arch",                  # 首要（archbishop, architect, archetype, archenemy）
    "auto",                  # 自己（automatic, autobiography, autonomous）
    "bene",                  # 好（benefit, benevolent, benediction, benefactor）
    "bi", "bin",             # 二（bicycle, binary, bilingual, binocular）
    "cata",                  # 向下（catalog, catastrophe, catabolism）
    "circum",                # 周围（circumference, circumstance, circumvent）
    "co", "col", "com", "con", "cor",  # 共同（cooperate, collect, combine, connect, correct）
    "contra", "counter",     # 反对（contradict, counteract, contravene）
    "de",                    # 向下/去除（decrease, deactivate, degrade, deconstruct）
    "deca",                  # 十（decade, decathlon, decalogue）
    "deci",                  # 十分之一（decimal, decimeter, decigram）
    "demi",                  # 半（demigod, demitasse, demilune）
    "di", "dis", "dif",      # 不/分开（differ, disable, disperse, diffuse, dissect）
    "dia",                   # 穿过（dialogue, diameter, diagnosis, diaphanous")
    "dys",                   # 坏的（dysfunction, dyslexia, dystopia, dyspepsia")
    "e", "ef",               # 出（evacuate, effect, eject, emanate, emerge）
    "em", "en",              # 使/进入（embody, enable, enclose, enact, empower")
    "endo",                  # 内（endoscopy, endogenous, endocrine）
    "epi",                   # 上/外（epidemic, epitome, epilogue, epidermis）
    "eu",                    # 好（euphoria, eulogy, eugenics, euthanasia")
    "ex", "exo",             # 出/外（export, exhale, exotic, exodus, exoskeleton）
    "extra", "extro",        # 以外（extraordinary, extrovert, extrapolate）
    "fore",                  # 前（forecast, foresee, foremost, forefather）
    "hemi",                  # 半（hemisphere, hemiplegia）
    "hetero",                # 异（heterodox, heterogeneous, heterosexual）
    "holo",                  # 全（hologram, holistic, holography）
    "homo",                  # 同（homogeneous, homonym, homogenize）
    "hyper",                 # 超过（hyperactive, hyperlink, hyperbole, hypertension）
    "hypo",                  # 下/低（hypothesis, hypodermic, hypothermia, hypoglycemia)
    "il", "im", "in", "ir",  # 不/进入（illegal, impossible, invisible, irregular, import）
    "infra",                 # 下（infrastructure, infrared, infrasonic）
    "inter",                 # 之间（internet, interact, international, interpret）
    "intra", "intro",        # 内部（intravenous, introduce, introspect, intranet）
    "iso",                   # 等（isotope, isometric, isomorphic）
    "macro",                 # 大（macroeconomics, macroscopic, macromolecule）
    "mal", "male",           # 坏（malfunction, maltreat, malice, malevolent, malnutrition)
    "meta",                  # 变化/超越（metaphor, metabolism, metacognition, metadata")
    "micro",                 # 小（microscope, microbe, microchip, microorganism）
    "mid",                   # 中（midnight, midway, midterm, midfield）
    "milli",                 # 千分之一（millimeter, milligram, millisecond）
    "mis",                   # 错误（mistake, misunderstand, misinform, misplace）
    "mono",                  # 单（monopoly, monotone, monologue, monochrome）
    "morph",                 # 形态（morphology, amorphous, metamorphosis）
    "multi",                 # 多（multiply, multimedia, multilingual, multitask）
    "myri",                  # 无数（myriad, myriapod）
    "neo",                   # 新（neonatal, neoclassical, neologism）
    "non",                   # 不（nonexistent, nonsense, nonverbal, nonprofit）
    "ob", "op",              # 对面/反（object, oppose, obstacle, obverse, oppress）
    "oct",                   # 八（octopus, octave, October, octagon）
    "omni",                  # 全（omnipotent, omniscient, omnivore, omnibus）
    "out",                   # 超过（outperform, outnumber, outline, outburst）
    "over",                  # 过度（overcome, overflow, overestimate, oversee）
    "paleo",                 # 古（paleolithic, paleontology）
    "pan", "panto",          # 全（panorama, pandemic, panacea, pantomime）
    "para",                  # 旁边/保护（paragraph, parallel, parasite, parachute）
    "pen",                   # 几乎（peninsula, penultimate, penumbra）
    "penta",                 # 五（pentagon, pentameter, pentathlon）
    "per",                   # 穿过/彻底（perfect, permeate, persist, perennial）
    "peri",                  # 周围（perimeter, peripheral, periscope, period）
    "poly",                  # 多（polygon, polyglot, polymer, polytechnic）
    "post",                  # 后（postpone, postwar, postmodern, posterior）
    "pre",                   # 前（predict, prefix, preliminary, precaution）
    "preter",                # 超过（preterit, preternatural, preterhuman）
    "pro",                   # 向前（produce, progress, promote, proactive）
    "proto",                 # 第一（prototype, protocol, protagonist, protozoa）
    "pseudo",                # 假（pseudonym, pseudoscience, pseudoevent）
    "quadr", "quart",        # 四（quadrilateral, quarter, quartet, quadratic）
    "quasi",                 # 准（quasi, quasijudicial, quasiofficial）
    "quint",                 # 五（quintuple, quintet, quintessential）
    "re",                    # 再次（return, review, rebuild, reverse）
    "retro",                 # 向后（retroactive, retrospect, retrograde, retrofit）
    "semi",                  # 半（semicircle, semifinal, semiconductor, semicolon）
    "sept",                  # 七（September, septet, septilateral）
    "sex",                   # 六（sextet, sextuple, sextant, sextuplet）
    "sub", "suc", "suf", "sup", "sus",  # 下（subway, succeed, suffix, support, suspend）
    "super", "supra",        # 上/超（superior, supernatural, supranational, supersede）
    "sur",                   # 上/超（surface, surpass, surcharge, surreal）
    "sym", "syn", "syl",     # 共同（symbol, synthesis, syllable, synonym, synergy）
    "tele",                  # 远（telephone, telescope, television, telegraph）
    "therm",                 # 热（thermal, thermometer, thermostat, thermodynamics）
    "trans",                 # 穿过（transport, transform, translate, transcontinental）
    "tri",                   # 三（triangle, trilogy, triathlon, trillion）
    "twi",                   # 二（twilight, twifold, twine）
    "ultra",                 # 超（ultraviolet, ultrasound, ultramodern, ultimate）
    "un",                    # 不（unhappy, unable, undo, unknown）
    "under",                 # 下/不足（understand, underground, underestimate, underlie）
    "uni",                   # 一（unique, uniform, universal, unicorn, unilateral）
    "up",                    # 上（upgrade, upright, uproot, uphold）
    "vice",                  # 副（vice-president, viceroy, vice versa）
    "xeno",                  # 异/外（xenon, xenophobia, xenophile）

    # ═══════════════════════════════════════════════════════════════
    # 第四部分：后缀（Suffix）
    # ═══════════════════════════════════════════════════════════════
    # 名词后缀
    "-tion", "-sion",        # 名词化（action, decision, education, expansion）
    "-ment",                 # 名词化（development, movement, government, element）
    "-ness",                 # 名词化（happiness, darkness, kindness, fitness）
    "-ity", "-ty",           # 性质（reality, ability, safety, diversity）
    "-ance", "-ence",        # 状态（resistance, difference, appearance, evidence）
    "-ism",                  # 主义/体系（capitalism, mechanism, organism, tourism）
    "-ist",                  # 人（scientist, artist, tourist, specialist）
    "-er", "-or",            # 人/物（teacher, doctor, computer, generator）
    "-age",                  # 集合/状态（storage, passage, damage, voltage）
    "-al",                   # 名词（proposal, arrival, survival, approval）
    "-dom",                  # 领域/状态（freedom, kingdom, wisdom, boredom）
    "-ful",                  # 充满（handful, cupful, spoonful）
    "-ship",                 # 身份/关系（friendship, leadership, membership, ownership）
    "-ure",                  # 状态/行为（failure, pressure, procedure, exposure）
    "-ics",                  # 学科（physics, mathematics, genetics, statistics）
    "-ology",                # 学科（biology, psychology, technology, methodology）
    "-graphy",               # 记录（geography, biography, photography, calligraphy）
    "-phobia",               # 恐惧（claustrophobia, arachnophobia, agoraphobia）
    "-cracy",                # 统治（democracy, bureaucracy, theocracy, aristocracy）
    "-archy",                # 统治（monarchy, anarchy, hierarchy, oligarchy）
    "-cide",                 # 杀（homicide, genocide, suicide, pesticide）
    "-icide",                # 杀（insecticide, herbicide, fungicide, biocide）

    # 形容词后缀
    "-able", "-ible",        # 能够（comfortable, possible, visible, readable）
    "-al",                   # 的（natural, original, chemical, musical）
    "-ful",                  # 充满（beautiful, powerful, wonderful, helpful）
    "-ous", "-ious",         # 充满（dangerous, mysterious, ambitious, curious）
    "-ive", "-ative",        # 性质（active, creative, massive, attractive）
    "-ic", "-ical",          # 的（scientific, historical, dramatic, chemical）
    "-less",                 # 无（homeless, careless, useless, wireless）
    "-ent", "-ant",          # 性质（different, important, excellent, current）
    "-ary", "-ory",          # 的（necessary, ordinary, laboratory, regulatory）
    "-esque",                # 风格（picturesque, Kafkaesque, grotesque, arabesque）
    "-like",                 # 像（lifelike, childlike, dreamlike, warlike）
    "-ward",                 # 方向（forward, backward, upward, outward）
    "-wise",                 # 方面（otherwise, likewise, clockwise, lengthwise）

    # 动词后缀
    "-ify",                  # 使（simplify, classify, verify, magnify, modify）
    "-ize", "-ise",          # 使（realize, organize, optimize, modernize, analyze）
    "-ate",                  # 使（activate, operate, generate, calculate, formulate）
    "-en",                   # 使（strengthen, widen, deepen, enlighten, broaden）
    "-esce",                 # 开始（coalesce, convalesce, acquiesce, evanesce）

    # 副词后缀
    "-ly",                   # 地（quickly, simply, exactly, relatively, literally）
    "-ward", "-wards",       # 方向（afterward, homeward, onwards, outwards）
    "-wise",                 # 方面（clockwise, otherwise, likewise, lengthwise）

    # ═══════════════════════════════════════════════════════════════
    # 第五部分：学术/专业补充词汇
    # ═══════════════════════════════════════════════════════════════
    # 通用学术
    "aforementioned", "hereinafter", "notwithstanding", "whereby",
    "therein", "thereof", "thereby", "thereafter", "heretofore",
    "inasmuch", "insofar", "nevertheless", "notwithstanding",
    "consequently", "furthermore", "moreover", "nevertheless",
    "subsequently", "predominantly", "fundamentally", "inherently",
    "comparatively", "unambiguously", "unequivocally", "systematically",
    "comprehensively", "methodologically", "theoretically", "empirically",
    "extrapolation", "interpolation", "approximation", "generalization",
    "simplification", "quantification", "normalization", "standardization",
    "optimization", "regularization", "differentiation", "integration",
    "discretization", "linearization", "orthogonalization", "factorization",
    "decomposition", "transformation", "canonicalization", "synchronization",
    # 数学/统计
    "eigen decomposition", "singular value", "Fourier transform",
    "Laplace transform", "Taylor series", "Maclaurin",
    "Cauchy", "Riemann", "Gaussian", "Bernoulli",
    "Markov", "Bayesian", "Frequentist", "stochastic",
    "deterministic", "probabilistic", "nonlinear", "parametric",
    "nonparametric", "asymptotic", "empirical", "theoretical",
    "discrete", "continuous", "convex", "concave",
    "monotonic", "bounded", "unbounded", "convergent",
    "divergent", "oscillatory", "analytic", "holomorphic",
    # 编程/机器学习
    "hyperparameter", "preprocessing", "postprocessing", "fine-tuning",
    "pretraining", "retraining", "transfer learning", "few-shot",
    "zero-shot", "self-supervised", "semi-supervised",
    "unsupervised", "reinforcement",
    "backpropagation", "gradient descent", "stochastic",
    "mini-batch", "cross-validation", "train-test split",
    "feature engineering", "dimensionality", "curse",
    "overfitting", "underfitting", "regularization",
    "normalization", "standardization", "augmentation",
    "initialization", "convergence", "divergence",
    # ═══════════════════════════════════════════════════════════════
    # 第六部分 字母
    # ═══════════════════════════════════════════════════════════════
    "a","b","c",
    "d","e","f","g","h",
    "i","g","k","l","m",
    "o","p","q","r","s",
    "t","u","v","w","x",
    "y","z","A","B","C",
    "D","E","F","G","H",
    "I","G","K","L","M",
    "O","P","Q","R","S",
    "T","U","V","W","X",
    "Y","Z"
]

CHEMISTRY_TOKENS = [
    # ── 常见分子式 ──
    "H2O", "CO2", "NH3", "CH4", "H2SO4", "HCl", "NaOH",
    "NaCl", "CaCO3", "C2H5OH", "C6H12O6", "HNO3", "H3PO4",
    "CH3COOH", "KMnO4", "Fe2O3", "Fe3O4", "CuSO4",
    "Ca(OH)2", "Mg(OH)2", "Al(OH)3", "NaHCO3", "Na2CO3",
    "H2O2", "SO2", "SO3", "NO2", "N2O", "CO", "H2S",
    "SiO2", "P2O5", "CaO", "MgO", "Al2O3", "ZnO", "CuO",
    "Na2SO4", "CaSO4", "BaSO4", "AgCl", "PbI2", "FeCl3",
    "NH4Cl", "NH4NO3", "(NH4)2SO4", "KClO3", "Ca3(PO4)2",
    "CH3OH", "C2H4", "C2H2", "C3H8", "C4H10", "C6H6",
    "C6H5OH", "C6H5CH3", "CH2O", "CCl4", "CHCl3",

    # ── 官能团 ──
    "methyl", "ethyl", "propyl", "butyl", "pentyl", "hexyl",
    "phenyl", "benzyl", "vinyl", "allyl", "acetyl",
    "hydroxyl", "carboxyl", "amino", "alkyl", "aryl",
    "carbonyl", "aldehyde", "ketone", "ester", "ether",
    "amide", "amine", "imine", "nitrile", "nitro",
    "sulfhydryl", "sulfone", "sulfoxide", "phosphate",
    "halide", "fluoride", "chloride", "bromide", "iodide",

    # ── 化学反应类型 ──
    "synthesis", "decomposition", "combustion", "neutralization",
    "precipitation", "displacement", "redox",
    "substitution", "addition", "elimination", "condensation",
    "hydrolysis", "polymerization", "esterification", "saponification",
    "oxidation", "reduction", "electrolysis", "catalysis",
    "isomerization", "racemization", "halogenation",
    "nitration", "sulfonation", "amination", "acetylation",
    "methylation", "alkylation", "acylation", "cyclization",

    # ── 化学键与分子结构 ──
    "covalent", "ionic", "metallic", "hydrogen bond",
    "van der Waals", "electronegativity", "valence",
    "hybridization", "orbital", "resonance", "conjugation",
    "stereoisomer", "enantiomer", "diastereomer", "chirality",
    "cis", "trans", "R-configuration", "S-configuration",
    "crystal", "amorphous", "lattice", "unit cell",

    # ── 化学分支 ──
    "organic", "inorganic", "analytical", "physical chemistry",
    "biochemistry", "electrochemistry", "thermochemistry",
    "photochemistry", "radiochemistry", "nuclear chemistry",
    "polymer", "colloid", "emulsion", "suspension",
    "solute", "solvent", "solution", "concentration",
    "molarity", "molality", "normality", "dilution",
    "titration", "indicator", "buffer", "pH",

    # ── 物质状态与性质 ──
    "solid", "liquid", "gas", "plasma",
    "melting", "boiling", "sublimation", "deposition",
    "evaporation", "condensation", "crystallization", "distillation",
    "viscosity", "density", "solubility", "conductivity",
    "flammable", "volatile", "corrosive", "toxic",
    "endothermic", "exothermic", "enthalpy", "entropy",
    "Gibbs", "Hess", "Le Chatelier", "Avogadro",

    # ── 元素与同位素 ──
    "isotope", "radioactive", "half-life", "decay",
    "alpha", "beta", "gamma",
    "periodic table", "alkali", "alkaline", "halogen",
    "noble gas", "transition metal", "lanthanide", "actinide",
    "metalloid", "nonmetal", "chalcogen", "pnictogen",

    # ── 元素符号（补全）──
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si",
    "Cl", "Ar", "Ca", "Sc", "Ti", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Zr",
    "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
]

候选列表 = []                         # 最终去重后的候选 token 列表
已加入 = set()                         # 已加入候选的 token 集合，用于 O(1) 去重
for t in LATEX_TOKENS + ENGLISH_TOKENS + CHEMISTRY_TOKENS:  # 按优先级顺序遍历（LaTeX > English > Chemistry）
    if t not in 已加入:                # 跳过重复 token（避免同一个 token 出现多次）
        候选列表.append(t)             # 加入候选列表
        已加入.add(t)                  # 标记为已加入


def 扩充分词器():                      # 主函数：从上游分词器读取词表，扩充到目标大小后写入输出目录
    assert 源分词器路径.exists(), f"源分词器不存在：{源分词器路径}"
    assert 目标词表大小 % 128 == 0, "目标词表大小必须是 128 的倍数"

    with open(源分词器路径, "r", encoding="utf-8") as f:  # 以 UTF-8 编码读取上游分词器文件
        tok_data = json.load(f)        # 将 JSON 解析为 Python 字典

    bpe_vocab: dict = tok_data["model"]["vocab"]  # BPE 词表：{token_str: id}
    added_tokens: list = tok_data.get("added_tokens", [])  # 已追加的特殊 token 列表

    已有token集 = set(bpe_vocab.keys())  # 从 BPE 词表收集所有已存在的 token 字符串
    for at in added_tokens:              # 同时把 added_tokens 中的 token 也加入集合
        已有token集.add(at["content"])   # 确保不会重复添加已存在的 token

    当前最大id = max(                     # 找出 BPE 词表和 added_tokens 中最大的 token id
        max(bpe_vocab.values()),          # BPE 词表中的最大 id
        max((at["id"] for at in added_tokens), default=-1)  # added_tokens 中的最大 id（可能为空列表）
    )
    当前总数 = 当前最大id + 1              # token id 从 0 开始，总数 = 最大 id + 1

    print(f"BPE 词表大小：{len(bpe_vocab)}")
    print(f"already_added_tokens：{len(added_tokens)}")
    print(f"当前最大 ID：{当前最大id}，推算词表总数：{当前总数}")
    print(f"目标词表大小：{目标词表大小}")

    需要新增数量 = 目标词表大小 - 当前总数
    assert 需要新增数量 > 0, (
        f"当前词表（{当前总数}）已超过目标（{目标词表大小}），请调大目标值"
    )
    print(f"需要新增：{需要新增数量} 个 token")

    待新增 = [t for t in 候选列表 if t not in 已有token集]
    print(f"候选新增（去重后）：{len(待新增)} 个")

    if len(待新增) < 需要新增数量:
        补齐数 = 需要新增数量 - len(待新增)
        print(f"候选不足，补充 {补齐数} 个占位 token")
        idx = 0
        while 补齐数 > 0:
            占位 = f"<ext_{idx:04d}>"
            if 占位 not in 已有token集 and 占位 not in set(待新增):
                待新增.append(占位)
                补齐数 -= 1
            idx += 1

    最终新增 = 待新增[:需要新增数量]        # 从候选列表中取前 N 个作为最终要添加的 token

    # Bug fix: 原代码只更新 added_tokens 列表，未同步 added_tokens_decoder 字典。
    # HuggingFace tokenizer.json 同时维护这两个字段，decode 新 token 时依赖
    # added_tokens_decoder（key 为 str(id)）。若缺失，加载时 decode 会失败或
    # 静默跳过这些 token，实际可用词表小于目标大小，训练中可能触发 embedding 越界。
    # 修复：在追加 added_tokens 的同时同步写入 added_tokens_decoder。
    if "added_tokens_decoder" not in tok_data:
        tok_data["added_tokens_decoder"] = {}

    下一个id = 当前最大id + 1              # 新 token 的起始 id，在现有最大 id 之后顺延
    for token_str in 最终新增:
        token_entry = {
            "id": 下一个id,
            "content": token_str,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": False,
        }
        added_tokens.append(token_entry)
        # 同步写入 added_tokens_decoder（key 必须为字符串形式的 id）
        tok_data["added_tokens_decoder"][str(下一个id)] = {
            k: v for k, v in token_entry.items() if k != "id"
        }
        下一个id += 1

    tok_data["added_tokens"] = added_tokens  # 将更新后的 added_tokens 列表写回 tokenizer 数据

    最终总数 = 当前最大id + 1 + len(最终新增)  # 原始词表大小 + 本次新增 token 数
    assert 最终总数 == 目标词表大小, f"词表大小异常：{最终总数} != {目标词表大小}"
    assert 最终总数 % 128 == 0, f"词表未对齐 128：{最终总数}"

    输出目录.mkdir(parents=True, exist_ok=True)  # 创建输出目录（已存在时不报错）
    输出路径 = 输出目录 / "tokenizer.json"         # 构建输出文件完整路径
    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, ensure_ascii=False, indent=2)
    print(f"已写出：{输出路径}")

    目标配置路径 = 输出目录 / "tokenizer_config.json"
    if 源配置路径.exists() and 源配置路径.resolve() != 目标配置路径.resolve():  # 源和目标不同文件时才复制
        shutil.copy2(源配置路径, 目标配置路径)  # copy2 保留文件元数据（修改时间等）
        print(f"已复制：tokenizer_config.json")

    latex_数 = sum(1 for t in 最终新增 if t in set(LATEX_TOKENS))    # 统计新增中 LaTeX token 的数量
    english_数 = sum(1 for t in 最终新增 if t in set(ENGLISH_TOKENS))  # 统计新增中英语 token 的数量
    chem_数 = sum(1 for t in 最终新增 if t in set(CHEMISTRY_TOKENS))   # 统计新增中化学 token 的数量
    placeholder_数 = len(最终新增) - latex_数 - english_数 - chem_数    # 剩余为占位 token
    print(f"\n新增明细：LaTeX {latex_数} | 英语 {english_数} | 化学 {chem_数} | 占位 {placeholder_数}")
    print(f"最终词表大小：{最终总数}（= 128 × {最终总数 // 128}）")


if __name__ == "__main__":             # 直接运行此脚本时执行分词器扩充
    扩充分词器()                        # 调用主函数完成词表扩充并写入磁盘
