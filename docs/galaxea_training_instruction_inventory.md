# Galaxea 训练指令原文与统计

真机实验请先看[按阶段排列的实验指令表](galaxea_training_instructions.md)。本页用于查找其他英文措辞和核对训练数据，不代表动作执行顺序。

范围：FastWAM、FastWAM-Joint 和 RIFT 共用的 445 条轨迹、379,564 帧，共 137 种精确英文文本（含异常字符串 `null`）。这些计数已按当前 RIFT 训练配置的 episode 过滤条件重新核对。

每帧的 `task_index` 对应同一采集批次 `meta/tasks.jsonl` 中的任务文本。当前配置选择 `中文@English` 的英文部分，仅使用当前动作指令，不拼接 high-level 文本或 `[Low]:`。原文保留标点、冠词和措辞差异。

下表帧数和轨迹数按该英文在全部八个采集批次中出现的情况统计；各章末尾的数据量则按采集批次统计。为便于核对 137 个唯一值，共用的开关水指令仅在水槽章列出；实验指令表中会在需要的流程重复列出。不同指令可能出现在同一条轨迹，因此逐行轨迹数不能相加。

## Clean The Sink

### 数据异常：`null`

> **异常：`null` 是字面字符串，不是缺失值。它实际进入了 39 个训练帧，涉及 38 条轨迹，不能作为正常真机指令。**

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `null` | 39 | 38 | null |

其完整 prompt 为：

```text
A video recorded from a robot's point of view executing the following instruction: null
```

### 打开水龙头

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文变体 |
|---|---:|---:|---|
| `Turn on the faucet with the left hand.` | 5,060 | 28 | 左手打开水龙头；左手将水龙头打开 |
| `Turn on the faucet with the right hand.` | 18,463 | 102 | 右手打开水龙头；右手将水龙头打开 |
| `Turn on the faucet with your left hand.` | 11,799 | 58 | 左手打开水龙头 |
| `Turn on the faucet with your right hand.` | 20,219 | 114 | 右手打开水龙头；右手将水龙头打开 |
| `Turn on the left faucet.` | 2,617 | 15 | 左手打开水龙头 |

### 关闭水龙头

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文变体 |
|---|---:|---:|---|
| `Turn off the faucet with the left hand.` | 7,254 | 48 | 左手关闭水龙头；左手将水龙头关闭 |
| `Turn off the faucet with the right hand.` | 20,044 | 148 | 右手关闭水龙头；右手将水龙头关闭 |
| `Turn off the faucet with your left hand.` | 4,328 | 32 | 左手关闭水龙头；左手将水龙头关闭 |
| `Turn off the faucet with your right hand.` | 4,052 | 34 | 右手关闭水龙头；右手将水龙头关闭 |

### 关闭水龙头并拿起毛巾

这些是数据中的单条复合命令，训练时没有被拆成两个 prompt。

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Close the faucet with left hand, then pick up the towel on the table.` | 1,456 | 7 | 左手关闭水龙头然后拿起桌面的毛巾 |
| `Close the faucet with the left hand, then pick up the towel on the table.` | 1,058 | 7 | 左手关闭水龙头然后拿起桌面的毛巾 |
| `Close the faucet with the right hand, then pick up the towel on the table.` | 1,247 | 8 | 右手关闭水龙头然后拿起桌面的毛巾 |
| `Close the faucet with your left hand, then pick up the towel on the table.` | 229 | 1 | 左手关闭水龙头然后拿起桌面的毛巾 |
| `Close the faucet with your right hand, then pick up the towel on the table.` | 1,161 | 7 | 右手关闭水龙头然后拿起桌面的毛巾 |
| `Turn off the faucet with the left hand, then pick up the towel on the table.` | 178 | 1 | 左手关闭水龙头然后拿起桌面的毛巾 |
| `Turn off the faucet with the right hand, then pick up the towel on the table.` | 2,408 | 15 | 右手关闭水龙头然后拿起桌面的毛巾 |

### 拿起抹布或毛巾

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the cloth on the sink with your left hand.` | 1,436 | 18 | 左手拿起洗手台上的抹布 |
| `Pick up the cloth on the sink with your right hand.` | 2,660 | 34 | 右手拿起洗手台上的抹布 |
| `Pick up the cloth on the washstand with your left hand.` | 325 | 4 | 左手拿起洗手台上的抹布 |
| `Pick up the towel on the table with your right hand.` | 94 | 1 | 右手拿起桌面的毛巾 |

### 擦拭或清洁水槽

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Clean the ink from the sink with a cloth using your left hand.` | 221 | 1 | 左手用抹布将洗手池的墨水擦干净 |
| `Clean the ink in the sink with a cloth using your left hand.` | 3,513 | 14 | 左手用抹布将洗手池的墨水擦干净 |
| `Clean the ink in the sink with a cloth using your right hand.` | 9,544 | 36 | 右手用抹布将洗手池的墨水擦干净 |
| `Clean the ink in the sink with the cloth using your left hand.` | 877 | 4 | 左手用抹布将洗手池的墨水擦干净 |
| `Use the towel with the left hand to wipe the sink.` | 1,795 | 8 | 左手使用毛巾擦拭洗手池 |
| `Use the towel with the right hand to wipe the sink.` | 2,248 | 12 | 右手使用毛巾擦拭洗手池 |
| `Use the towel with your left hand to wipe the sink.` | 1,391 | 6 | 左手使用毛巾擦拭洗手池 |
| `Use the towel with your right hand to wipe the sink.` | 3,210 | 19 | 右手使用毛巾擦拭洗手池 |
| `Wipe the ink off the sink with a cloth using your left hand.` | 614 | 3 | 左手用抹布将洗手池的墨水擦干净 |

### 放回抹布或毛巾

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Place the cloth back on the left side of the sink with the left hand.` | 117 | 1 | 左手将抹布放回洗手台左侧 |
| `Place the cloth back on the left side of the sink with your left hand.` | 2,390 | 19 | 左手将抹布放回洗手台左侧 |
| `Place the cloth back on the right side of the sink with the right hand.` | 153 | 1 | 右手将抹布放回洗手台右侧 |
| `Place the cloth back on the right side of the sink with your right hand.` | 3,690 | 33 | 右手将抹布放回洗手台右侧 |
| `Place the cloth back to the left side of the sink with your left hand.` | 98 | 1 | 左手将抹布放回洗手台左侧 |
| `Place the cloth on the right side of the sink with your right hand.` | 111 | 1 | 右手将抹布放回洗手台右侧 |
| `Place the towel back on the table with the left hand.` | 107 | 1 | 左手将毛巾放回桌面 |
| `Place the towel back on the table with the right hand.` | 1,144 | 10 | 右手将毛巾放回桌面 |
| `Place the towel back on the table with your left hand.` | 1,988 | 14 | 左手将毛巾放回桌面 |
| `Place the towel back on the table with your right hand.` | 2,578 | 21 | 右手将毛巾放回桌面 |

### 拿布、擦拭并归位的复合命令

这些也是训练中的完整单条 prompt，不能按句号或 `and/then` 拆分。

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the cloth with the right hand, wipe the sink with the cloth using the right hand, and return the cloth to its original place with the right hand.` | 7,603 | 12 | 右手拿起抹布，右手使用抹布擦拭洗手池，右手将抹布放回原处 |
| `Pick up the cloth with the right hand, wipe the sink with the cloth using the right hand, and return the cloth to its place with the right hand.` | 522 | 1 | 右手拿起抹布，右手使用抹布擦拭洗手池，右手将抹布放回原处 |
| `Pick up the cloth with the right hand, wipe the sink with the cloth using the right hand, then return the cloth to its original place with the right hand.` | 6,567 | 11 | 右手拿起抹布，右手使用抹布擦拭洗手池，右手将抹布放回原处 |
| `Pick up the cloth with the right hand. Wipe the sink with the cloth using the right hand. Return the cloth to its original place with the right hand.` | 30,584 | 54 | 右手拿起抹布，右手使用抹布擦拭洗手池，右手将抹布放回原处 |
| `Pick up the cloth with the right hand. Wipe the sink with the cloth using the right hand. Return the cloth to its original position with the right hand.` | 643 | 1 | 右手拿起抹布，右手使用抹布擦拭洗手池，右手将抹布放回原处 |

### 数据范围与来源

以下三个水槽采集批次合计 **183 条轨迹 / 142,861 帧 / 45 种英文文本**。这是采集批次的统计；上方逐条指令的计数包含它在其他任务中的出现次数。

| collection | retained episodes | rows | collection 内唯一文本数 |
|---|---:|---:|---:|
| `Clean_The_Sink_20250710_006` | 79 | 70,342 | 10 |
| `Clean_The_Sink_20250721_008` | 58 | 42,863 | 22 |
| `Clean_The_Sink_20250730_012` | 46 | 29,656 | 22 |

collection 内唯一数不能直接相加，因为多个 collection 使用了相同的英文文本。通用 faucet 文本也出现在部分 Cutting Board 数据中；为保证 137 个全局唯一值各出现一次，它们统一归在本章。

## Clean The Cutting Board

本章合并两个原始 coarse labels：`clean cutting board` 与 `clean the cutting board`。两者只反映采集批次的 high-level 命名差异，均未进入训练 prompt；下表不合并、不润色其 low-level English。

### 保持机身不动

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Body remain stationary.` | 63 | 2 | 本体未动 |
| `Do not move the body.` | 86 | 2 | 本体未动 |
| `Do not move.` | 251 | 7 | 本体未动 |
| `Keep body stationary.` | 336 | 8 | 本体未动 |
| `Remain stationary.` | 1,059 | 23 | 本体未动 |

### 打开水龙头

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Turn on the left tap.` | 366 | 3 | 左手打开水龙头 |

其他与 Sink 完全同文的通用 faucet 指令只在 `Clean The Sink` 章列一次。

### 拿起案板

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the cutting board with the right hand.` | 2,575 | 10 | 右手拿起案板 |

### 拿起海绵百洁布

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the sponge scouring pad with the left hand.` | 5,017 | 31 | 左手拿起海绵百洁布 |
| `Pick up the sponge scouring pad with your left hand.` | 933 | 6 | 左手拿起海绵百洁布 |
| `Pick up the sponge with the left hand.` | 2,875 | 19 | 左手拿起海绵百洁布 |

### 擦拭或清洁案板

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Clean the cutting board with the sponge scouring pad using the left hand.` | 3,364 | 14 | 左手用海绵百洁布擦拭案板 |
| `Clean the cutting board with the sponge scouring pad using your left hand.` | 11,772 | 41 | 左手用海绵百洁布擦拭案板 |
| `Clean the cutting board with the sponge side of the scrubber using your left hand.` | 177 | 1 | 左手用海绵百洁布擦拭案板 |
| `Use the left hand to wipe the cutting board with a sponge scouring pad.` | 486 | 3 | 左手用海绵百洁布擦拭案板 |
| `Use the left hand with a sponge scouring pad to wipe the cutting board.` | 163 | 1 | 左手用海绵百洁布擦拭案板 |
| `Use the sponge scouring pad in your left hand to wipe the cutting board.` | 868 | 4 | 左手用海绵百洁布擦拭案板 |
| `Use the sponge scouring pad with the left hand to wipe the cutting board.` | 1,524 | 7 | 左手用海绵百洁布擦拭案板 |
| `Use the sponge scouring pad with your left hand to wipe the cutting board.` | 1,445 | 6 | 左手用海绵百洁布擦拭案板 |

### 冲洗、托举或冲洗后归位案板

这些文本按原始命令保留；包含 `pick up`、`hold`、`rinse/wash`、`return/place` 的行都是单条复合训练输入。

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Hold the cutting board under the faucet with your left hand.` | 2,965 | 12 | 左手拿起案板举在出水口下 |
| `Hold the cutting board under the water outlet with your left hand.` | 8,541 | 36 | 左手拿起案板举在出水口下 |
| `Pick up the cutting board with your left hand and rinse it under the faucet.` | 4,503 | 19 | 左手拿起案板放到水龙头下冲洗 |
| `Place the cutting board under the faucet with left hand and rinse, then return it to the table.` | 271 | 1 | 左手将案板放置于水龙头下冲洗，将案板放回桌面 |
| `Place the cutting board under the faucet with the left hand and rinse it, then return it to the table.` | 736 | 4 | 左手将案板放置于水龙头下冲洗，将案板放回桌面 |
| `Place the cutting board under the faucet with the left hand and rinse it, then return the cutting board to the table.` | 1,782 | 8 | 左手将案板放置于水龙头下冲洗，将案板放回桌面 |
| `Place the cutting board under the faucet with the left hand to rinse, then return it to the table.` | 351 | 1 | 左手将案板放置于水龙头下冲洗，将案板放回桌面 |
| `Place the cutting board under the faucet with your left hand and rinse it, then return it to the table.` | 494 | 2 | 左手将案板放置于水龙头下冲洗，将案板放回桌面 |
| `Place the cutting board under the faucet with your left hand and rinse it.` | 225 | 1 | 左手将案板放置于水龙头下冲洗 |
| `Place the cutting board under the faucet with your left hand to rinse, then return it to the table.` | 828 | 4 | 左手将案板放置于水龙头下冲洗，将案板放回桌面 |
| `Rinse the cutting board with the right hand and place it back on the sink countertop.` | 320 | 1 | 右手将案板冲洗后放回洗手池台面 |
| `Rinse the cutting board with the right hand, then place it back on the sink countertop.` | 274 | 1 | 右手将案板冲洗后放回洗手池台面 |
| `Rinse the cutting board with the right hand.` | 25,955 | 56 | 右手将案板进行冲洗 |
| `Rinse the chopping board clean with the left hand, then place it on the table.` | 3,255 | 12 | 左手将案板冲洗干净后放到桌面上 |
| `Rinse the chopping board with left hand and place it on the table.` | 1,044 | 4 | 左手将案板冲洗干净后放到桌面上 |
| `Rinse the chopping board with the left hand and place it on the table.` | 1,740 | 6 | 左手将案板冲洗干净后放到桌面上 |
| `Rinse the chopping board with the left hand, then place it on the table.` | 2,118 | 9 | 左手将案板冲洗干净后放到桌面上 |
| `Rinse the cutting board clean with the left hand, then place it on the table.` | 329 | 1 | 左手将案板冲洗干净后放到桌面上 |
| `Rinse the cutting board with left hand and place it on the table.` | 654 | 3 | 左手将案板冲洗干净后放到桌面上 |
| `Rinse the cutting board with the left hand and place it on the table.` | 335 | 2 | 左手将案板冲洗干净后放到桌面上 |
| `Rinse the cutting board with the left hand, then place it on the table.` | 121 | 1 | 左手将案板冲洗干净后放到桌面上 |
| `Wash the chopping board with the left hand and place it on the table.` | 299 | 1 | 左手将案板冲洗干净后放到桌面上 |
| `Wash the cutting board with the left hand and place it on the table.` | 608 | 2 | 左手将案板冲洗干净后放到桌面上 |
| `Wash the cutting board with the left hand, then place it on the table.` | 1,644 | 9 | 左手将案板冲洗干净后放到桌面上 |

### 放回案板或海绵百洁布

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Place the cutting board back on the sink countertop with the right hand.` | 1,561 | 7 | 右手将案板放回洗手池台面 |
| `Place the cutting board back on the sink countertop with your right hand.` | 12,231 | 48 | 右手将案板放回洗手池台面 |
| `Place the sponge scouring pad back on the sink countertop with the left hand.` | 630 | 8 | 左手将海绵百洁布放回洗手池台面 |
| `Place the sponge scouring pad back on the sink countertop with your left hand.` | 3,833 | 48 | 左手将海绵百洁布放回洗手池台面 |

### 拿起海绵、清洁案板并归位的复合命令

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the sponge scouring pad with the right hand and wipe the cutting board, then return the sponge scouring pad to its original position with the right hand.` | 4,477 | 13 | 右手拿起海绵百洁布将案板擦拭，右手将海绵百洁布放回原位 |
| `Pick up the sponge scouring pad with the right hand and wipe the cutting board. Return the sponge scouring pad to its original position with the right hand.` | 893 | 3 | 右手拿起海绵百洁布将案板擦拭，右手将海绵百洁布放回原位 |
| `Pick up the sponge scouring pad with the right hand, clean the cutting board, then place it on the table.` | 11,329 | 25 | 右手拿起海绵百洁布清洗案板然后放到桌子上 |
| `Pick up the sponge scouring pad with your right hand, clean the cutting board, and place it on the table.` | 940 | 2 | 右手拿起海绵百洁布清洗案板然后放到桌子上 |
| `Pick up the sponge scouring pad with your right hand, clean the cutting board, and then place it on the table.` | 748 | 1 | 右手拿起海绵百洁布清洗案板然后放到桌子上 |
| `Pick up the sponge scouring pad with your right hand, clean the cutting board, then place it on the table.` | 2,606 | 6 | 右手拿起海绵百洁布清洗案板然后放到桌子上 |
| `Pick up the sponge scrubber with the right hand and wipe the cutting board, then return the sponge scrubber to its original position with the right hand.` | 530 | 2 | 右手拿起海绵百洁布将案板擦拭，右手将海绵百洁布放回原位 |
| `Pick up the sponge scrubber with the right hand and wipe the cutting board, then return the sponge scrubber to its place with the right hand.` | 682 | 2 | 右手拿起海绵百洁布将案板擦拭，右手将海绵百洁布放回原位 |
| `Pick up the sponge scrubber with the right hand, clean the cutting board, then place it on the table.` | 353 | 1 | 右手拿起海绵百洁布清洗案板然后放到桌子上 |
| `Pick up the sponge with the right hand, clean the cutting board, then place it on the table.` | 5,932 | 13 | 右手拿起海绵百洁布清洗案板然后放到桌子上 |
| `Pick up the sponge with the right hand, scrub the cutting board, then place it on the table.` | 544 | 1 | 右手拿起海绵百洁布清洗案板然后放到桌子上 |

### 数据范围与来源

以下三个案板采集批次合计 **127 条轨迹 / 185,014 帧**。本章列出 57 种英文文本；与水槽共用的英文已在水槽章列出。

| collection | coarse label | retained episodes | rows | collection 内唯一文本数 |
|---|---|---:|---:|---:|
| `Clean_Cutting_Board_20250627_002` | `clean cutting board` | 58 | 100,733 | 32 |
| `Clean_The_Cutting_Board_20250711_006` | `clean the cutting board` | 20 | 22,238 | 16 |
| `Clean_The_Cutting_Board_20250804_012` | `clean the cutting board` | 49 | 62,043 | 25 |

collection 内唯一文本数包含与 Sink 完全相同的通用 faucet 文本；这些字符串只在 `Clean The Sink` 章列一次。因此该列不能相加得到本章的 57。

## Wipe The Desktop

### 拿起抹布

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the cloth with left hand.` | 1,045 | 12 | 左手拿起抹布 |
| `Pick up the cloth with the left hand.` | 2,991 | 35 | 左手拿起抹布 |
| `Pick up the cloth with the right hand.` | 4,624 | 56 | 右手拿起抹布 |
| `Pick up the cloth with your left hand.` | 673 | 8 | 左手拿起抹布 |

### 拿起抹布并放到桌面

这些是单条复合训练输入，不等同于分别发送 `pick up` 和 `place`。

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the cloth with left hand and place it on the table.` | 842 | 4 | 左手拿起抹布，放到桌面上 |
| `Pick up the cloth with the left hand and place it on the table.` | 1,890 | 7 | 左手拿起抹布，放到桌面上 |
| `Pick up the cloth with the right hand and place it on the table.` | 1,696 | 9 | 右手拿起抹布，放到桌面上 |
| `Pick up the cloth with your left hand and place it on the table.` | 1,573 | 8 | 左手拿起抹布，放到桌面上 |
| `Pick up the cloth with your right hand and place it on the table.` | 1,989 | 10 | 右手拿起抹布，放到桌面上 |

### 擦拭或清洁桌面污渍

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Clean the stain on the table with the cloth in your left hand.` | 129 | 1 | 左手使用抹布擦拭桌面上的污渍 |
| `Clean the stain on the table with the cloth using the left hand.` | 128 | 1 | 左手使用抹布擦拭桌面上的污渍 |
| `Clean the stain on the table with the cloth using your left hand.` | 3,367 | 22 | 左手使用抹布擦拭桌面上的污渍 |
| `Use the cloth in your left hand to wipe the stain on the table.` | 212 | 1 | 左手使用抹布擦拭桌面上的污渍 |
| `Use the cloth with the right hand to wipe the stain on the table.` | 267 | 2 | 右手使用抹布擦拭桌面上的污渍 |
| `Use the cloth with your left hand to wipe the stain on the table.` | 1,139 | 10 | 左手使用抹布擦拭桌面上的污渍 |
| `Use the cloth with your left hand to wipe the stains on the table.` | 1,150 | 7 | 左手使用抹布擦拭桌面上的污渍 |
| `Use the cloth with your right hand to wipe the stain on the table.` | 1,198 | 8 | 右手使用抹布擦拭桌面上的污渍 |
| `Use the cloth with your right hand to wipe the stains on the table.` | 534 | 4 | 右手使用抹布擦拭桌面上的污渍 |
| `Use the left hand to wipe the stain on the table with a cloth.` | 1,337 | 11 | 左手使用抹布擦拭桌面上的污渍 |
| `Use the left hand to wipe the stains on the table with a cloth.` | 234 | 1 | 左手使用抹布擦拭桌面上的污渍 |
| `Use the left hand with a cloth to wipe the stain on the table.` | 108 | 2 | 左手使用抹布擦拭桌面上的污渍 |
| `Use the right hand to wipe the stain on the table with a cloth.` | 2,313 | 17 | 右手使用抹布擦拭桌面上的污渍 |
| `Use the right hand to wipe the stains on the table with a cloth.` | 95 | 1 | 右手使用抹布擦拭桌面上的污渍 |
| `Use the right hand with a cloth to wipe the stain on the table.` | 3,075 | 22 | 右手使用抹布擦拭桌面上的污渍 |
| `Use the right hand with a cloth to wipe the stains on the table.` | 210 | 1 | 右手使用抹布擦拭桌面上的污渍 |

### 将抹布放到桌面

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Place the cloth on the table with left hand.` | 81 | 1 | 左手将抹布放到桌面上 |
| `Place the cloth on the table with the left hand.` | 1,552 | 18 | 左手将抹布放到桌面上 |
| `Place the cloth on the table with the right hand.` | 2,710 | 29 | 右手将抹布放到桌面上 |
| `Place the cloth on the table with your left hand.` | 4,177 | 37 | 左手将抹布放到桌面上 |
| `Place the cloth on the table with your right hand.` | 2,112 | 26 | 右手将抹布放到桌面上 |

### 拿起毛巾、擦拭并放回的复合命令

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the towel with left hand and wipe the stain, then place the towel on the table with left hand.` | 631 | 2 | 左手拿起毛巾擦拭污渍，左手把毛巾放到桌子上 |
| `Pick up the towel with the left hand and wipe the stain, then place the towel on the table with the left hand.` | 2,391 | 8 | 左手拿起毛巾擦拭污渍，左手把毛巾放到桌子上 |
| `Pick up the towel with the right hand and wipe the stain, then place the towel on the table with the right hand.` | 4,468 | 14 | 右手拿起毛巾擦拭污渍，右手把毛巾放到桌子上 |
| `Pick up the towel with the right hand and wipe the stain. Place the towel on the table with the right hand.` | 349 | 1 | 右手拿起毛巾擦拭污渍，右手把毛巾放到桌子上 |
| `Pick up the towel with your right hand and wipe the stain, then place the towel on the table with your right hand.` | 398 | 1 | 右手拿起毛巾擦拭污渍，右手把毛巾放到桌子上 |

### 数据范围与来源

以下两个桌面采集批次合计 **135 条轨迹 / 51,689 帧 / 35 种英文文本**。

| collection | coarse label | retained episodes | rows | collection 内唯一文本数 |
|---|---|---:|---:|---:|
| `Wipe_The_Desktop_20250801_012` | `wipe the desktop` | 26 | 8,237 | 5 |
| `Wipe_The_Desktop_20250804_011` | `wipe the desktop` | 109 | 43,452 | 31 |

两个 collection 有一个相同文本，因此 collection 内唯一数之和为 36，而全局并集为 35。

真机操作阶段及输入格式见[实验指令表](galaxea_training_instructions.md)。

全局口径：三个任务合计 **445 episodes / 379,564 rows / 137 个实际唯一 bare-English low-level instruction**；其中 `Clean The Sink` 45 个、`Clean The Cutting Board` 57 个、`Wipe The Desktop` 35 个。
