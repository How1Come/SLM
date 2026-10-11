"""Project-original modern Chinese web-novel register (网文体), license-clear.

Procedural scene bank — not scraped from paywalled fiction sites.
"""
from __future__ import annotations

import itertools
import random

SYSTEM = (
    "你是这场互动小说的叙述者。用中文第三人称描写场景与动作，并写出至少两名有名字的角色对白。"
    "对白用「」标注，说话前写明角色名。适当推进情节，不要跳出故事解释规则。"
)

SETTINGS = [
    ("深夜的写字楼三十一层，加班灯还亮着", "都市职场"),
    ("电竞馆决赛后台，粉丝欢呼声隔着隔音门传来", "电竞"),
    ("灵气复苏后的地铁换乘站，安检门闪着淡蓝符文", "都市异能"),
    ("宗门外门试炼场，石碑上浮现本轮任务", "修仙"),
    ("刑侦大队询问室，监控红灯静静亮着", "悬疑"),
    ("恋爱综艺录制间隙，化妆间里只剩两个人", "综艺恋综"),
    ("末日避难所通风管道口，应急灯忽明忽暗", "末世"),
    ("海边民宿暴雨夜，Wi-Fi 断了只剩雨声", "悬疑情感"),
    ("直播平台顶层办公室，热搜词条还在跳", "直播"),
    ("高校社团招新棚下，传单被风吹得乱响", "青春"),
]

NAMES = [
    ("林晚", "周澈"), ("苏棠", "陆沉"), ("陈野", "韩栀"), ("顾言", "宋乔"),
    ("唐栗", "许珩"), ("赵澄", "何渺"), ("方琪", "梁策"), ("叶鸣", "秦雾"),
    ("贺川", "阮宁"), ("程曜", "白露"), ("沈星", "罗野"), ("姜禾", "季扬"),
]

HOOKS = [
    "手机里跳出一条匿名消息，只有一句话和一个倒计时。",
    "门外传来本该不存在于此的脚步声。",
    "桌上的证物袋被人动过，封条却完好。",
    "直播弹幕突然齐刷刷刷起同一个名字。",
    "灵力检测仪嗡鸣，数值直接冲破安全线。",
    "综艺耳返里导播的声音发颤：别按原计划走。",
    "避难所广播插播一条未备案的撤离口令。",
    "对方递来的合同最后一页，签名是空的。",
]

BEATS = [
    ("摊牌", "一方要求对方今晚之前给出明确选择。"),
    ("交换条件", "一方提出交换情报，另一方要求先看到筹码。"),
    ("第三方介入", "第三人推门而入，气氛骤然收紧。"),
    ("误会澄清", "关键物证出现，先前判断被推翻。"),
    ("险中求进", "两人决定先假意配合，再反手破局。"),
]


def _response(setting: str, a: str, b: str, hook: str, beat_name: str, beat: str) -> str:
    return (
        f"{setting}。{hook}\n\n"
        f"{a}抬眼打量{b}，沉声道：「今晚这事，你我都别装糊涂。」\n\n"
        f"{b}握紧手机，低声答：「我可以配合，但你得先告诉我——那条消息是谁发的？」\n\n"
        f"{a}苦笑：「我也想知道。」他顿了顿，{beat} {b}与{a}对视，同时意识到：{beat_name}只是开始。"
    )


def iter_modern_wen_examples(n: int, seed: int = 0) -> list[dict]:
    random.seed(seed)
    combos = list(
        itertools.product(range(len(SETTINGS)), range(len(NAMES)), range(len(HOOKS)), range(len(BEATS)))
    )
    random.shuffle(combos)
    out: list[dict] = []
    for si, ni, hi, bi in combos[:n]:
        setting, tag = SETTINGS[si]
        a, b = NAMES[ni]
        hook = HOOKS[hi]
        beat_name, beat = BEATS[bi]
        user = (
            f"【{tag}】{setting}。{a}与{b}在场。请用网文常见的现代白话节奏续写这一幕，"
            f"写出至少两人的对白并推进情节。"
        )
        out.append(
            {
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                "response": _response(setting, a, b, hook, beat_name, beat),
                "_src": f"modern-wen:{tag}",
                "_license": "project-original-modern-wen",
            }
        )
    return out
