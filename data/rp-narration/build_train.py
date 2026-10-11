#!/usr/bin/env python3
"""Build train.jsonl for narration LoRA (original Chinese scenes)."""
import json
import os
import random

SYSTEM = (
    "你是这场互动小说的叙述者。用中文第三人称描写场景与动作，并写出至少两名有名字的角色对白。"
    "对白用「」标注，说话前写明角色名。适当推进情节，不要跳出故事解释规则。"
)

# Original scene bank (project-authored; PD-style situations only).
SCENES = [
    (
        "雨夜，客栈里只有一盏油灯。掌柜老周和赶路的书生李言正在说话。请续写这一幕。",
        "油灯噼啪一声，灯芯爆出一点火星。窗外雨声如织，檐角水珠连成线，砸在青石板上。\n\n"
        "老周把抹布搭在肩上，抬眼打量李言湿透的袖口，叹道：「这位相公，再往前走二十里也没有第二家客栈了。」\n\n"
        "李言抱紧书箱，苦笑：「正因如此，我才想问问，方才楼上那阵脚步，是不是贵店还有别的住客？」\n\n"
        "老周神色一滞，压低声音：「相公别多问。雨停之前，谁也别上楼。」话音未落，楼梯口又传来一声轻响，"
        "仿佛有人赤足踏过。李言与老周对视一眼，油灯的影子在墙上猛地晃了一下。",
    ),
    (
        "山道上，猎户大牛与药娘小芹发现了受伤的狐狸。请续写。",
        "晨雾未散，山道泥泞。大牛背着猎弓蹲下身，见草叶上血迹点点；小芹已取出金疮药，把狐狸轻轻翻了过来。\n\n"
        "大牛皱眉：「这种毛色，不像寻常野狐。」\n\n"
        "小芹指尖一颤：「它后腿被夹子所伤，却不见猎户记号。大牛哥，别动它的项圈。」\n\n"
        "狐狸忽然睁眼，目光竟似人言，嘶哑开口：「二位若救我，三日之内，莫向村里人说见过我。」\n\n"
        "大牛握紧弓弦，小芹却已把药粉敷上。两人尚未答话，林深处传来一声哨响，惊起满林飞鸟。",
    ),
    (
        "城门口，守将韩彪与来投军的少年阿木对峙。请续写。",
        "雪风卷着旌旗，城门铁甲铿锵。韩彪按刀而立，俯视眼前瘦小的少年，冷声问：「你说你父亲在军中？」\n\n"
        "阿木挺直背脊，从怀里掏出半块锈蚀军牌：「韩将军，我爹叫阿柱，三年前守北门战死。我来替他完成未竟的誓言。」\n\n"
        "韩彪接过军牌，指节发白，良久才道：「军牌是真的。可你未满十六，按律不能入伍。」\n\n"
        "阿木咬牙：「那我便在城外立棚，直到将军肯收我。」韩彪抬眼望向城楼上飘落的雪，忽然转身对副将道："
        "「先带他去伙房。明日卯时，校场见。」阿木一愣，韩彪已低声补了一句：「别让你爹的名字蒙尘。」",
    ),
    (
        "月下荷塘边，琴师云舒与侍女阿巧说起了旧事。请续写。",
        "月色铺在水面，荷香随风。云舒横抱古琴，指尖仍颤着余音；阿巧捧着披风，站在亭外不敢上前。\n\n"
        "云舒轻声道：「阿巧，你还记得三年前那场火吗？」\n\n"
        "阿巧眼眶一红：「奴婢记得。您从火里抱出琴匣，手都烧伤了，却一句疼也没喊。」\n\n"
        "云舒苦笑：「那夜丢的不止琴社，还有一个人答应要听我的新曲。」她抬眸望向远处城楼，"
        "「明日若有人带桃花来寻，你只管让他进，不必通报。」阿巧刚要应声，荷塘里忽地一声水响，"
        "仿佛有人潜在莲叶之下。",
    ),
    (
        "雪后的集市上，铁匠老石与卖糖人的小满起了争执。请续写。",
        "雪光映得摊位发亮。老石挥舞铁钳，怒道：「你那糖签子扔在我煤炉边，差点引火！」\n\n"
        "小满也不示弱：「你那铁屑溅到我糖浆里，半锅糖全废了，谁赔？」\n\n"
        "围观人群渐多。老石闷哼一声，从怀里摸出几枚铜钱；小满却忽然收声，盯着他袖口："
        "「你袖口的血……不是铁锈。」老石神色骤变，拽着小满往巷后走，低声道：「别嚷。昨晚有人闯进铺子，"
        "我只砍中他一刀。」小满倒吸凉气：「那你更该报官！」老石摇头：「报官之前，得先抓住来取刀的人。」",
    ),
    (
        "渡口暮色里，船夫老杜与戴斗笠的客人对坐。请续写。",
        "江水呜咽，缆绳轻轻拍打船帮。老杜划着桨，斜眼打量客人：「客官要去对岸黑松林？」\n\n"
        "客人摘下斗笠，露出半张烧伤的脸：「是。我知道你不载生人夜渡，但我带的是救命的药。」\n\n"
        "老杜沉默片刻，把桨一横：「药给谁？」客人道：「给林子里一个等死的孩子。他爹当年救过我的命。」\n\n"
        "老杜啐了一口，却调转船头：「上舱坐稳。若见水鬼拖桨，别伸手。」客人刚要道谢，舱底传来孩童的咳嗽，"
        "老杜猛地回头——船上竟还有第三个人。",
    ),
    (
        "药铺后堂，掌柜孙仁与学徒阿禾核对账本。请续写。",
        "药柜抽屉半开，甘苦气味交织。孙仁敲着算盘，忽然停住：「少了一味三七，昨夜谁动过？」\n\n"
        "阿禾脸色发白：「师父，我昨夜只煎了前堂李婆的药，没进后柜。」\n\n"
        "孙仁眯眼：「李婆的药方里本无三七。」他起身往后院走，阿禾紧跟。两人推开柴门，见地上有新鲜鞋印，"
        "一路通向围墙缺口。孙仁低声道：「阿禾，去前堂稳住客人。」阿禾急道：「您呢？」孙仁已从架上抽下一包石灰："
        "「我去会会这位不请自来的‘药贼’。」",
    ),
    (
        "古寺钟楼下，游方僧慧明与打更人老槐说话。请续写。",
        "残雪挂在檐角，铜钟默然。慧明拂去石阶尘土，问：「更夫，昨夜钟响三声，可是你在？」\n\n"
        "老槐颤声答：「大师，我昨夜发寒疾，并未出门。钟……是自个儿响的。」\n\n"
        "慧明抬头望钟，轻声念咒，忽然钟内传来女子叹息。老槐吓得跌坐，慧明却道：「女施主，若有所冤，"
        "可随我入殿暂栖。」话音甫落，钟楼阴影里走出一名抱琴女子，衣角尚滴着水。老槐惊呼，女子已向慧明一揖："
        "「多谢师父。我被人沉在井中，只求借钟音，唤我兄长来听。」",
    ),
]


def paraphrase_opening(user: str) -> str:
    """Light user-side variants so the model generalizes beyond exact prompts."""
    swaps = [
        ("请续写这一幕。", "请接着写下去。"),
        ("请续写。", "请继续这一幕。"),
        ("请续写", "往下写"),
    ]
    out = user
    for a, b in swaps:
        if a in out:
            out = out.replace(a, b, 1)
            break
    return out


def expand_scenes():
    rows = []
    for user, response in SCENES:
        rows.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "response": response})
        alt = paraphrase_opening(user)
        if alt != user:
            rows.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": alt}], "response": response})
    # Extra short prompts sharing the same responses (teaches name + dialogue pattern)
    extras = [
        ("茶馆里，说书人老谭与听客少年小陆起了一点争执。请续写。", None),
        ("边关烽火台，老兵秦川与传令兵小马核对军情。请续写。", None),
        ("渔村晒网场，渔妇阿莲与货郎金福讨价还价。请续写。", None),
    ]
    for user, _ in extras:
        # reuse rotated responses with light prefix adaptation
        base = random.choice(SCENES)[1]
        rows.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "response": base})
    return rows


def template_scenes(n=36):
    """Synthetic but original micro-scenes (name pairs rotated)."""
    places = ["客栈", "渡口", "药铺", "擂台", "雪岭小庙", "渔村晒场", "城门口", "荷塘亭"]
    pairs = [
        ("老周", "李言"), ("大牛", "小芹"), ("韩彪", "阿木"), ("云舒", "阿巧"), ("老石", "小满"),
        ("孙仁", "阿禾"), ("慧明", "老槐"), ("刘安", "小虎"), ("铁臂梁", "阿铜"), ("阿雪", "楚帆"),
    ]
    rows = []
    for i in range(n):
        place = places[i % len(places)]
        a, b = pairs[i % len(pairs)]
        user = f"{place}里，{a}与{b}正在交谈。请续写这一幕。"
        response = (
            f"{place}风声渐紧。{a}抬眼打量对方，沉声道：「今日这事，你我都别声张。」\n\n"
            f"{b}握紧袖口，低声答：「我答应你，但你也得告诉我，那封信究竟给谁？」\n\n"
            f"{a}苦笑：「给一个不该还活着的人。」话音未落，门外脚步杂乱，有人高声喊着{a}的名字。"
            f"{b}与{a}对视，同时伸手按住了桌上的刀与伞。"
        )
        rows.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "response": response})
    return rows


def more_variants():
    """Additional wholly original micro-scenes."""
    bank = [
        (
            "夜深，驿馆天井里，驿丞刘安与驿卒小虎看见白衣女子飘过。请续写。",
            "灯笼晃了晃，白衣一闪而没入回廊。小虎牙齿打战：「头儿，那……那没脚声。」\n\n"
            "刘安按住他肩：「别嚷。去把东厢的客人叫醒，就说驿馆走水，请他们移去西厢。」\n\n"
            "小虎不解：「没走水啊？」刘安盯着回廊尽头：「比走水更麻烦。」他话音刚落，东厢传来婴孩啼哭，"
            "而登记册上，东厢本为空房。",
        ),
        (
            "擂台边，拳师铁臂梁与师弟阿铜商量退赛。请续写。",
            "台下人声鼎沸。梁用布条缠紧拳锋，低声道：「今日对手来路不明，你不能上场。」\n\n"
            "阿铜急道：「师兄，咱们师门名声……」梁打断：「名声换不得你的命。」他从怀里取出一枚黑棋："
            "「方才有人塞我这东西，说是‘输一局，保一命’。」阿铜还要争辩，擂台鼓声已响，对面那人缓缓抬头，"
            "面具下竟与梁有七分相像。",
        ),
        (
            "雪岭小庙，猎户少女阿雪与伤兵楚帆分食最后一块饼。请续写。",
            "柴火噼啪，饼香微弱。阿雪把饼掰成两半，较大的一块推给楚帆：「你是朝廷的人，得留着力气。」\n\n"
            "楚帆摇头：「姑娘救我三日，该是我欠你。」他忽然咳出黑血，又强撑笑容，「若我活不过今夜，"
            "帮我把这枚印信送到关内。」阿雪接过印信，触手冰凉。庙门外风雪骤紧，有人高声问：「里面可有逃兵？」"
            "楚帆与阿雪对视，同时握紧了柴刀与猎弓。",
        ),
    ]
    return [
        {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}], "response": r}
        for u, r in bank
    ]


def main():
    random.seed(0)
    rows = expand_scenes() + more_variants() + template_scenes(12)
    # duplicate core scenes with second system hint (SillyTavern-like world card snippet)
    world = SYSTEM + "\n\n【世界】架空古代，允许轻微志怪元素，但人物动机要清楚。"
    for user, response in SCENES[:5]:
        rows.append({"messages": [{"role": "system", "content": world}, {"role": "user", "content": user}], "response": response})
    out = os.path.join(os.path.dirname(__file__), "train.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples -> {out}")


if __name__ == "__main__":
    main()
