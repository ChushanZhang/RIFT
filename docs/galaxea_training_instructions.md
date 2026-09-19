# Galaxea 真机实验指令表

按任务、操作手和工具摆放选择下面的一套流程。表中的英文均为训练原句，可以直接复制为当前阶段的 instruction；左右手以机器人自身为准。

1. 从与现场状态对应的阶段开始，发送该行完整英文。
2. 观察动作完成后，再换下一行；最后一行完成后结束该流程。
3. 一行包含多个动作时，整段保持同一条 instruction，例如“拿布、擦拭、放回”完成后才切换。

切换条件是供实验员判断的完成标准，不是程序自动检测，也不按训练视频的秒数定时切换。只复制英文，不添加中文、`[Low]:` 或任务标题；部署端使用的完整文本格式见[末尾说明](#输入格式与数据依据)。

| 任务 | 选择流程 |
|---|---|
| [清洁水槽](#清洁水槽) | 抹布分步执行；抹布连续执行；毛巾（关水与拿取合为一条） |
| [清洁案板](#清洁案板) | 右手持板、左手擦；左手持板、右手擦（两种采集流程） |
| [擦桌面](#擦桌面) | 单手分步执行；先换手再擦；毛巾连续执行 |

## 清洁水槽

这三套训练流程都是**先开水、再关水，随后拿布或毛巾擦拭**。

### 抹布：分步执行

场景：清理水槽墨水。右手流程把抹布放在水槽右侧台面，左手流程放在左侧；擦完放回同侧。整套选择右手或左手。来源：`Clean_The_Sink_20250721_008`。

**右手**

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 开水 | `Turn on the faucet with the right hand.` | 水龙头已打开 |
| 2. 关水 | `Turn off the faucet with your right hand.` | 水龙头已关闭 |
| 3. 拿抹布 | `Pick up the cloth on the sink with your right hand.` | 右手已抓稳抹布，可进入擦拭 |
| 4. 擦拭墨水 | `Clean the ink in the sink with a cloth using your right hand.` | 目标墨水已擦除 |
| 5. 放回抹布 | `Place the cloth back on the right side of the sink with your right hand.` | 抹布已放回洗手台右侧并松开，结束 |

**左手**

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 开水 | `Turn on the faucet with the left hand.` | 水龙头已打开 |
| 2. 关水 | `Turn off the faucet with the left hand.` | 水龙头已关闭 |
| 3. 拿抹布 | `Pick up the cloth on the sink with your left hand.` | 左手已抓稳抹布，可进入擦拭 |
| 4. 擦拭墨水 | `Clean the ink in the sink with a cloth using your left hand.` | 目标墨水已擦除 |
| 5. 放回抹布 | `Place the cloth back on the left side of the sink with your left hand.` | 抹布已放回洗手台左侧并松开，结束 |

### 抹布：拿取、擦拭、放回连续执行

场景：抹布放在水槽右侧台面。右手操作水龙头，再用同一只手完成整段擦拭，最后把布放回右侧。来源：`Clean_The_Sink_20250710_006`。

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 开水 | `Turn on the faucet with your right hand.` | 水龙头已打开 |
| 2. 关水 | `Turn off the faucet with the right hand.` | 水龙头已关闭 |
| 3. 拿抹布、擦水槽、放回 | `Pick up the cloth with the right hand. Wipe the sink with the cloth using the right hand. Return the cloth to its original place with the right hand.` | 擦拭完成，抹布已放回原处并松开，结束 |

第 3 行中的三个句子是一条 instruction，不在拿起抹布后换成另一条。

### 毛巾：关水和拿取合为一条

场景：毛巾放在水龙头旁的水槽台面，右手流程放右侧，左手流程放左侧。关水后拿起，擦完放回同侧。英文原句中的 `table` 在这套示范中指该处台面。来源：`Clean_The_Sink_20250730_012`。

**右手**

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 开水 | `Turn on the faucet with your right hand.` | 水龙头已打开 |
| 2. 关水并拿毛巾 | `Turn off the faucet with the right hand, then pick up the towel on the table.` | 水龙头已关闭，右手已拿起毛巾 |
| 3. 擦水槽 | `Use the towel with your right hand to wipe the sink.` | 目标区域完成擦拭 |
| 4. 放回毛巾 | `Place the towel back on the table with the right hand.` | 毛巾已放回同侧台面并松开，结束 |

**左手**

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 开水 | `Turn on the faucet with the left hand.` | 水龙头已打开 |
| 2. 关水并拿毛巾 | `Close the faucet with left hand, then pick up the towel on the table.` | 水龙头已关闭，左手已拿起毛巾 |
| 3. 擦水槽 | `Use the towel with the left hand to wipe the sink.` | 目标区域完成擦拭 |
| 4. 放回毛巾 | `Place the towel back on the table with your left hand.` | 毛巾已放回同侧台面并松开，结束 |

<details>
<summary>另一个训练流程：右手分开关水和拿毛巾</summary>

该批次的 `episode_000013` 将这两个动作分别标注。若实验采用这套分步方式，使用下面的完整流程。

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 开水 | `Turn on the faucet with your right hand.` | 水龙头已打开 |
| 2. 关水 | `Turn off the faucet with the right hand.` | 水龙头已关闭 |
| 3. 拿毛巾 | `Pick up the towel on the table with your right hand.` | 右手已拿起毛巾 |
| 4. 擦水槽 | `Use the towel with your right hand to wipe the sink.` | 目标区域完成擦拭 |
| 5. 放回毛巾 | `Place the towel back on the table with the right hand.` | 毛巾已放回桌面并松开，结束 |

</details>

## 清洁案板

先确定**哪只手持案板、哪只手拿海绵**，整套按对应流程执行。下列三套保留各自采集批次的顺序与原句。

### 右手持板，左手拿海绵

场景：案板放在水槽右侧台面，海绵放在左侧。来源：`Clean_Cutting_Board_20250627_002`。下表对应两轮“冲洗 → 擦拭”的完整示范；冲洗时右手持板，擦拭时仍由右手持板、左手用海绵清洁。

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 左手开水 | `Turn on the faucet with your left hand.` | 水龙头已打开 |
| 2. 左手拿海绵 | `Pick up the sponge scouring pad with the left hand.` | 左手已拿起海绵 |
| 3. 右手拿板并冲洗 | `Rinse the cutting board with the right hand.` | 右手已持板完成本轮冲洗，并将板面转向海绵 |
| 4. 左手擦案板 | `Clean the cutting board with the sponge scouring pad using your left hand.` | 完成本轮擦拭，进入下一轮冲洗 |
| 5. 右手再次冲洗 | `Rinse the cutting board with the right hand.` | 完成第二轮冲洗 |
| 6. 左手再次擦拭 | `Clean the cutting board with the sponge scouring pad using your left hand.` | 完成第二轮擦拭，进入收尾 |
| 7. 右手放回案板 | `Place the cutting board back on the sink countertop with the right hand.` | 案板已放到洗手池台面并松开 |
| 8. 左手放回海绵 | `Place the sponge scouring pad back on the sink countertop with your left hand.` | 海绵已放到洗手池台面并松开 |
| 9. 左手关水 | `Turn off the faucet with the left hand.` | 水龙头已关闭，结束 |

其他轨迹也有一轮或三轮版本；实验前确定轮数，冲洗与擦拭保持交替。画面确认本表示范的拿板动作包含在第 3 行 `Rinse...` 指令段内，无需另外插入拿板指令。

<details>
<summary>单独发送“拿起案板”的分步流程</summary>

该批次有 10 条轨迹单独标注拿板动作。下面使用 `episode_000074` 的原句；其中的“本体未动”标注不作为操作阶段。

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 左手开水 | `Turn on the faucet with your left hand.` | 水龙头已打开 |
| 2. 左手拿海绵 | `Pick up the sponge with the left hand.` | 左手已拿起海绵 |
| 3. 右手抓住案板 | `Pick up the cutting board with the right hand.` | 右手已夹住案板，随后切换到冲洗；不必等板完全抬起 |
| 4. 右手抬板并冲洗 | `Rinse the cutting board with the right hand.` | 右手已抬板并完成本轮冲洗 |
| 5. 左手擦案板 | `Clean the cutting board with the sponge scouring pad using your left hand.` | 完成本轮擦拭 |
| 6. 右手再次冲洗 | `Rinse the cutting board with the right hand.` | 完成第二轮冲洗 |
| 7. 左手再次擦拭 | `Clean the cutting board with the sponge scouring pad using your left hand.` | 完成第二轮擦拭 |
| 8. 右手放回案板 | `Place the cutting board back on the sink countertop with the right hand.` | 案板已放到洗手池台面并松开 |
| 9. 左手放回海绵 | `Place the sponge scouring pad back on the sink countertop with your left hand.` | 海绵已放到洗手池台面并松开 |
| 10. 左手关水 | `Turn off the faucet with the left hand.` | 水龙头已关闭，结束 |

该示范的拿板标注结束时，案板仍在台面附近；抬起动作在下一条冲洗指令下继续完成。

</details>

### 左手持板，右手擦拭后把海绵放回原位

场景：案板放在水槽左侧台面，海绵放在右侧。来源：`Clean_The_Cutting_Board_20250711_006`。左手负责案板的拿取、冲洗和放回；右手负责开关水及拿海绵擦拭。

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 右手开水 | `Turn on the faucet with the right hand.` | 水龙头已打开 |
| 2. 左手拿板并冲洗 | `Pick up the cutting board with your left hand and rinse it under the faucet.` | 左手已拿起案板并完成初次冲洗 |
| 3. 右手拿海绵、擦板、放回海绵 | `Pick up the sponge scouring pad with the right hand and wipe the cutting board, then return the sponge scouring pad to its original position with the right hand.` | 擦拭完成，海绵已放回原位；左手仍持案板 |
| 4. 左手再冲洗并放回案板 | `Place the cutting board under the faucet with the left hand and rinse it, then return the cutting board to the table.` | 案板已冲洗并放回桌面，左手松开 |
| 5. 右手关水 | `Turn off the faucet with the right hand.` | 水龙头已关闭，结束 |

### 左手持板，右手擦拭后把海绵放到台面

场景：案板放在水槽左侧台面，海绵放在右侧。来源：`Clean_The_Cutting_Board_20250804_012`。这一批的英文措辞与上一套不同，保留整套原句。

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 右手开水 | `Turn on the faucet with your right hand.` | 水龙头已打开 |
| 2. 左手把案板举到出水口下 | `Hold the cutting board under the water outlet with your left hand.` | 左手持案板，案板已到出水口下 |
| 3. 右手拿海绵、擦板、放下海绵 | `Pick up the sponge scouring pad with the right hand, clean the cutting board, then place it on the table.` | 海绵已放到水槽右侧台面；左手仍持案板 |
| 4. 左手冲洗并放回案板 | `Rinse the chopping board clean with the left hand, then place it on the table.` | 案板已冲洗并放到桌面，左手松开 |
| 5. 右手关水 | `Turn off the faucet with the right hand.` | 水龙头已关闭，结束 |

第 3 行的 `it` 在这条示范中指海绵，画面可见海绵回到右侧台面；案板仍由左手拿着，到第 4 行才放回。

## 擦桌面

### 抹布：单手分步执行

场景：抹布已在擦拭手可拿取的位置。示范中，右手流程从桌面右侧拿布，左手流程从左侧拿布。来源：`Wipe_The_Desktop_20250804_011`。

**右手**

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 拿抹布 | `Pick up the cloth with the right hand.` | 右手已抓住并拿起抹布 |
| 2. 擦污渍 | `Use the right hand to wipe the stain on the table with a cloth.` | 目标污渍已擦除 |
| 3. 放下抹布 | `Place the cloth on the table with the right hand.` | 抹布已放到桌面并松开，结束 |

**左手**

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 拿抹布 | `Pick up the cloth with the left hand.` | 左手已抓住并拿起抹布 |
| 2. 擦污渍 | `Clean the stain on the table with the cloth using your left hand.` | 目标污渍已擦除 |
| 3. 放下抹布 | `Place the cloth on the table with the left hand.` | 抹布已放到桌面并松开，结束 |

### 抹布：先用另一只手放到桌面，再拿起擦拭

场景：抹布先由另一只手拿取，放到擦拭手一侧的桌面，再由擦拭手拿起。例如右手准备、左手擦时，布从右侧转放到左侧。两只手通过桌面转移抹布，不是直接在手中交接。来源仍为 `Wipe_The_Desktop_20250804_011`。

**右手准备，左手擦**

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 右手拿布并放到桌面 | `Pick up the cloth with your right hand and place it on the table.` | 右手已把抹布放到桌面并松开 |
| 2. 左手拿布 | `Pick up the cloth with the left hand.` | 左手已拿起抹布 |
| 3. 左手擦污渍 | `Use the left hand to wipe the stain on the table with a cloth.` | 目标污渍已擦除 |
| 4. 左手放下布 | `Place the cloth on the table with your left hand.` | 抹布已放到桌面并松开，结束 |

**左手准备，右手擦**

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 左手拿布并放到桌面 | `Pick up the cloth with left hand and place it on the table.` | 左手已把抹布放到桌面并松开 |
| 2. 右手拿布 | `Pick up the cloth with the right hand.` | 右手已拿起抹布 |
| 3. 右手擦污渍 | `Use the right hand to wipe the stain on the table with a cloth.` | 目标污渍已擦除 |
| 4. 右手放下布 | `Place the cloth on the table with the right hand.` | 抹布已放到桌面并松开，结束 |

<details>
<summary>少数训练轨迹中的其他安排</summary>

`episode_000024`、`episode_000090` 先完成一套左手“拿布 → 擦拭 → 放下”，再完成一套右手流程。`episode_000063` 先用左手拿布放到桌面，随后仍由左手重新拿起、擦拭、放下。需要复现这些安排时，原始标注可按末尾路径回查。

</details>

### 毛巾：拿取、擦拭、放回连续执行

来源：`Wipe_The_Desktop_20250801_012`。毛巾放在所用手一侧的桌面。按所用手选择一条，整段执行期间保持这句 instruction；完成后毛巾仍留在同侧桌面，不要求精确回到原点。

**右手**

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 右手拿毛巾、擦污渍、放回 | `Pick up the towel with your right hand and wipe the stain, then place the towel on the table with your right hand.` | 擦拭完成，毛巾已放到桌面并松开，结束 |

**左手**

| 阶段 | 发送的 instruction（整句复制） | 什么时候换下一条 |
|---|---|---|
| 1. 左手拿毛巾、擦污渍、放回 | `Pick up the towel with the left hand and wipe the stain, then place the towel on the table with the left hand.` | 擦拭完成，毛巾已放到桌面并松开，结束 |

## 输入格式与数据依据

上述英文是 instruction 字段的内容。RIFT 训练时的完整文本为：

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

`{task}` 替换成表中的整句英文。真机部署端应与此保持一致；若客户端已有此模板，不要在输入框重复添加。当前仓库没有可核实的 Galaxea 真机客户端，本文不假定客户端会自动加前缀。

阶段顺序已按[训练配置](../configs/data/galaxea_indoor_cleaning.yaml)的 episode 过滤条件，核对服务器上的 **445 条轨迹、379,564 帧**。英文指令逐字来自每套标明的代表轨迹，未做同义改写。另抽查了 **15 条代表轨迹各阶段的起始、中间、结束画面**，核对工具摆放、持物手和放回对象。

这些核对不代表真机成功率测试，也不能保证每个标注边界都恰好对应动作完成。表中的水流停止、污渍擦除等条件仍由实验员现场确认。

完整的 **137 种训练文本、中文和出现次数**见[文末折叠附录](#完整指令与统计)。其中 `null` 是异常标注，不作为操作指令；“本体未动”一类文本也不作为流程步骤或机器人暂停按钮。

<details>
<summary>代表轨迹与帧范围（供回查训练数据）</summary>

数据根目录：`/cfsdata/chushan/Galaxea-Dataset/manipulation-clean/`。

每条轨迹的指令来自 `<collection>/data/chunk-000/episode_<六位编号>.parquet` 的 `task_index`，在同目录的 `meta/tasks.jsonl` 中查原文。头部视频位于 `<collection>/videos/chunk-000/observation.images.head_rgb/episode_<六位编号>.mp4`。下列范围首尾均包含，视频为 15 fps。

帧范围仅用于定位示范，不能作为真机自动切换时间。下表除 `Clean_The_Sink_20250730_012/episode_000013` 只核对了标注外，其余 15 条均已抽查阶段画面。

| 采集批次 | 代表轨迹 | 表中各阶段的帧范围 |
|---|---|---|
| `Clean_The_Sink_20250721_008` | `episode_000000` | 0–196 → 197–291 → 292–389 → 390–966 → 967–1109 |
| `Clean_The_Sink_20250721_008` | `episode_000030` | 0–204 → 205–302 → 303–388 → 389–703 → 704–798 |
| `Clean_The_Sink_20250710_006` | `episode_000000` | 0–187 → 188–340 → 341–914 |
| `Clean_The_Sink_20250730_012` | `episode_000000` | 0–177 → 178–367 → 368–516 → 517–606 |
| `Clean_The_Sink_20250730_012` | `episode_000021` | 0–174 → 175–322 → 323–507 → 508–672 |
| `Clean_The_Sink_20250730_012` | `episode_000013` | 0–164 → 165–288 → 289–382 → 383–462 → 463–553 |
| `Clean_Cutting_Board_20250627_002` | `episode_000073` | 0–192 → 193–330 → 331–671 → 672–836 → 837–893 → 894–1070 → 1071–1280 → 1281–1339 → 1340–1446 |
| `Clean_Cutting_Board_20250627_002` | `episode_000074` | 0–197 → 198–323 → 360–512 → 513–898 → 899–1074 → 1075–1198 → 1199–1360 → 1361–1613 → 1614–1684 → 1685–1832 |
| `Clean_The_Cutting_Board_20250711_006` | `episode_000001` | 0–168 → 169–351 → 352–740 → 741–975 → 976–1116 |
| `Clean_The_Cutting_Board_20250804_012` | `episode_000027` | 0–103 → 104–256 → 257–638 → 639–904 → 905–1069 |
| `Wipe_The_Desktop_20250804_011` | `episode_000000` | 0–80 → 81–202 → 203–267 |
| `Wipe_The_Desktop_20250804_011` | `episode_000002` | 0–134 → 135–236 → 237–346 |
| `Wipe_The_Desktop_20250804_011` | `episode_000001` | 0–262 → 263–359 → 360–514 → 515–635 |
| `Wipe_The_Desktop_20250804_011` | `episode_000025` | 0–247 → 248–322 → 323–419 → 420–537 |
| `Wipe_The_Desktop_20250801_012` | `episode_000000` | 0–397 |
| `Wipe_The_Desktop_20250801_012` | `episode_000013` | 0–351 |

这些流程来自完整代表轨迹。少数轨迹有 `null`、缺少阶段标注或前后手标注不一致；本文没有把这些差异补写成新的动作顺序。完整词表保留这些训练文本，便于排查。

</details>

## 完整指令与统计

<details>
<summary>展开全部 137 条训练原文、中文和统计</summary>

以下内容用于查找其他英文措辞和核对训练数据，不代表动作执行顺序。

范围：FastWAM、FastWAM-Joint 和 RIFT 共用的 445 条轨迹、379,564 帧，共 137 种精确英文文本（含异常字符串 `null`）。这些计数已按当前 RIFT 训练配置的 episode 过滤条件重新核对。

每帧的 `task_index` 对应同一采集批次 `meta/tasks.jsonl` 中的任务文本。当前配置选择 `中文@English` 的英文部分，仅使用当前动作指令，不拼接 high-level 文本或 `[Low]:`。原文保留标点、冠词和措辞差异。

下表帧数和轨迹数按该英文在全部八个采集批次中出现的情况统计；各章末尾的数据量则按采集批次统计。为便于核对 137 个唯一值，共用的开关水指令仅在水槽章列出；实验指令表中会在需要的流程重复列出。不同指令可能出现在同一条轨迹，因此逐行轨迹数不能相加。

### Clean The Sink

#### 数据异常：`null`

> **异常：`null` 是字面字符串，不是缺失值。它实际进入了 39 个训练帧，涉及 38 条轨迹，不能作为正常真机指令。**

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `null` | 39 | 38 | null |

其完整 prompt 为：

```text
A video recorded from a robot's point of view executing the following instruction: null
```

#### 打开水龙头

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文变体 |
|---|---:|---:|---|
| `Turn on the faucet with the left hand.` | 5,060 | 28 | 左手打开水龙头；左手将水龙头打开 |
| `Turn on the faucet with the right hand.` | 18,463 | 102 | 右手打开水龙头；右手将水龙头打开 |
| `Turn on the faucet with your left hand.` | 11,799 | 58 | 左手打开水龙头 |
| `Turn on the faucet with your right hand.` | 20,219 | 114 | 右手打开水龙头；右手将水龙头打开 |
| `Turn on the left faucet.` | 2,617 | 15 | 左手打开水龙头 |

#### 关闭水龙头

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文变体 |
|---|---:|---:|---|
| `Turn off the faucet with the left hand.` | 7,254 | 48 | 左手关闭水龙头；左手将水龙头关闭 |
| `Turn off the faucet with the right hand.` | 20,044 | 148 | 右手关闭水龙头；右手将水龙头关闭 |
| `Turn off the faucet with your left hand.` | 4,328 | 32 | 左手关闭水龙头；左手将水龙头关闭 |
| `Turn off the faucet with your right hand.` | 4,052 | 34 | 右手关闭水龙头；右手将水龙头关闭 |

#### 关闭水龙头并拿起毛巾

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

#### 拿起抹布或毛巾

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the cloth on the sink with your left hand.` | 1,436 | 18 | 左手拿起洗手台上的抹布 |
| `Pick up the cloth on the sink with your right hand.` | 2,660 | 34 | 右手拿起洗手台上的抹布 |
| `Pick up the cloth on the washstand with your left hand.` | 325 | 4 | 左手拿起洗手台上的抹布 |
| `Pick up the towel on the table with your right hand.` | 94 | 1 | 右手拿起桌面的毛巾 |

#### 擦拭或清洁水槽

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

#### 放回抹布或毛巾

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

#### 拿布、擦拭并归位的复合命令

这些也是训练中的完整单条 prompt，不能按句号或 `and/then` 拆分。

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the cloth with the right hand, wipe the sink with the cloth using the right hand, and return the cloth to its original place with the right hand.` | 7,603 | 12 | 右手拿起抹布，右手使用抹布擦拭洗手池，右手将抹布放回原处 |
| `Pick up the cloth with the right hand, wipe the sink with the cloth using the right hand, and return the cloth to its place with the right hand.` | 522 | 1 | 右手拿起抹布，右手使用抹布擦拭洗手池，右手将抹布放回原处 |
| `Pick up the cloth with the right hand, wipe the sink with the cloth using the right hand, then return the cloth to its original place with the right hand.` | 6,567 | 11 | 右手拿起抹布，右手使用抹布擦拭洗手池，右手将抹布放回原处 |
| `Pick up the cloth with the right hand. Wipe the sink with the cloth using the right hand. Return the cloth to its original place with the right hand.` | 30,584 | 54 | 右手拿起抹布，右手使用抹布擦拭洗手池，右手将抹布放回原处 |
| `Pick up the cloth with the right hand. Wipe the sink with the cloth using the right hand. Return the cloth to its original position with the right hand.` | 643 | 1 | 右手拿起抹布，右手使用抹布擦拭洗手池，右手将抹布放回原处 |

#### 数据范围与来源

以下三个水槽采集批次合计 **183 条轨迹 / 142,861 帧 / 45 种英文文本**。这是采集批次的统计；上方逐条指令的计数包含它在其他任务中的出现次数。

| collection | retained episodes | rows | collection 内唯一文本数 |
|---|---:|---:|---:|
| `Clean_The_Sink_20250710_006` | 79 | 70,342 | 10 |
| `Clean_The_Sink_20250721_008` | 58 | 42,863 | 22 |
| `Clean_The_Sink_20250730_012` | 46 | 29,656 | 22 |

collection 内唯一数不能直接相加，因为多个 collection 使用了相同的英文文本。通用 faucet 文本也出现在部分 Cutting Board 数据中；为保证 137 个全局唯一值各出现一次，它们统一归在本章。

### Clean The Cutting Board

本章合并两个原始 coarse labels：`clean cutting board` 与 `clean the cutting board`。两者只反映采集批次的 high-level 命名差异，均未进入训练 prompt；下表不合并、不润色其 low-level English。

#### 保持机身不动

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Body remain stationary.` | 63 | 2 | 本体未动 |
| `Do not move the body.` | 86 | 2 | 本体未动 |
| `Do not move.` | 251 | 7 | 本体未动 |
| `Keep body stationary.` | 336 | 8 | 本体未动 |
| `Remain stationary.` | 1,059 | 23 | 本体未动 |

#### 打开水龙头

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Turn on the left tap.` | 366 | 3 | 左手打开水龙头 |

其他与 Sink 完全同文的通用 faucet 指令只在 `Clean The Sink` 章列一次。

#### 拿起案板

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the cutting board with the right hand.` | 2,575 | 10 | 右手拿起案板 |

#### 拿起海绵百洁布

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the sponge scouring pad with the left hand.` | 5,017 | 31 | 左手拿起海绵百洁布 |
| `Pick up the sponge scouring pad with your left hand.` | 933 | 6 | 左手拿起海绵百洁布 |
| `Pick up the sponge with the left hand.` | 2,875 | 19 | 左手拿起海绵百洁布 |

#### 擦拭或清洁案板

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

#### 冲洗、托举或冲洗后归位案板

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

#### 放回案板或海绵百洁布

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Place the cutting board back on the sink countertop with the right hand.` | 1,561 | 7 | 右手将案板放回洗手池台面 |
| `Place the cutting board back on the sink countertop with your right hand.` | 12,231 | 48 | 右手将案板放回洗手池台面 |
| `Place the sponge scouring pad back on the sink countertop with the left hand.` | 630 | 8 | 左手将海绵百洁布放回洗手池台面 |
| `Place the sponge scouring pad back on the sink countertop with your left hand.` | 3,833 | 48 | 左手将海绵百洁布放回洗手池台面 |

#### 拿起海绵、清洁案板并归位的复合命令

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

#### 数据范围与来源

以下三个案板采集批次合计 **127 条轨迹 / 185,014 帧**。本章列出 57 种英文文本；与水槽共用的英文已在水槽章列出。

| collection | coarse label | retained episodes | rows | collection 内唯一文本数 |
|---|---|---:|---:|---:|
| `Clean_Cutting_Board_20250627_002` | `clean cutting board` | 58 | 100,733 | 32 |
| `Clean_The_Cutting_Board_20250711_006` | `clean the cutting board` | 20 | 22,238 | 16 |
| `Clean_The_Cutting_Board_20250804_012` | `clean the cutting board` | 49 | 62,043 | 25 |

collection 内唯一文本数包含与 Sink 完全相同的通用 faucet 文本；这些字符串只在 `Clean The Sink` 章列一次。因此该列不能相加得到本章的 57。

### Wipe The Desktop

#### 拿起抹布

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the cloth with left hand.` | 1,045 | 12 | 左手拿起抹布 |
| `Pick up the cloth with the left hand.` | 2,991 | 35 | 左手拿起抹布 |
| `Pick up the cloth with the right hand.` | 4,624 | 56 | 右手拿起抹布 |
| `Pick up the cloth with your left hand.` | 673 | 8 | 左手拿起抹布 |

#### 拿起抹布并放到桌面

这些是单条复合训练输入，不等同于分别发送 `pick up` 和 `place`。

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the cloth with left hand and place it on the table.` | 842 | 4 | 左手拿起抹布，放到桌面上 |
| `Pick up the cloth with the left hand and place it on the table.` | 1,890 | 7 | 左手拿起抹布，放到桌面上 |
| `Pick up the cloth with the right hand and place it on the table.` | 1,696 | 9 | 右手拿起抹布，放到桌面上 |
| `Pick up the cloth with your left hand and place it on the table.` | 1,573 | 8 | 左手拿起抹布，放到桌面上 |
| `Pick up the cloth with your right hand and place it on the table.` | 1,989 | 10 | 右手拿起抹布，放到桌面上 |

#### 擦拭或清洁桌面污渍

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

#### 将抹布放到桌面

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Place the cloth on the table with left hand.` | 81 | 1 | 左手将抹布放到桌面上 |
| `Place the cloth on the table with the left hand.` | 1,552 | 18 | 左手将抹布放到桌面上 |
| `Place the cloth on the table with the right hand.` | 2,710 | 29 | 右手将抹布放到桌面上 |
| `Place the cloth on the table with your left hand.` | 4,177 | 37 | 左手将抹布放到桌面上 |
| `Place the cloth on the table with your right hand.` | 2,112 | 26 | 右手将抹布放到桌面上 |

#### 拿起毛巾、擦拭并放回的复合命令

| 英文指令原文 | 帧数 | 轨迹数 | 原始中文 |
|---|---:|---:|---|
| `Pick up the towel with left hand and wipe the stain, then place the towel on the table with left hand.` | 631 | 2 | 左手拿起毛巾擦拭污渍，左手把毛巾放到桌子上 |
| `Pick up the towel with the left hand and wipe the stain, then place the towel on the table with the left hand.` | 2,391 | 8 | 左手拿起毛巾擦拭污渍，左手把毛巾放到桌子上 |
| `Pick up the towel with the right hand and wipe the stain, then place the towel on the table with the right hand.` | 4,468 | 14 | 右手拿起毛巾擦拭污渍，右手把毛巾放到桌子上 |
| `Pick up the towel with the right hand and wipe the stain. Place the towel on the table with the right hand.` | 349 | 1 | 右手拿起毛巾擦拭污渍，右手把毛巾放到桌子上 |
| `Pick up the towel with your right hand and wipe the stain, then place the towel on the table with your right hand.` | 398 | 1 | 右手拿起毛巾擦拭污渍，右手把毛巾放到桌子上 |

#### 数据范围与来源

以下两个桌面采集批次合计 **135 条轨迹 / 51,689 帧 / 35 种英文文本**。

| collection | coarse label | retained episodes | rows | collection 内唯一文本数 |
|---|---|---:|---:|---:|
| `Wipe_The_Desktop_20250801_012` | `wipe the desktop` | 26 | 8,237 | 5 |
| `Wipe_The_Desktop_20250804_011` | `wipe the desktop` | 109 | 43,452 | 31 |

两个 collection 有一个相同文本，因此 collection 内唯一数之和为 36，而全局并集为 35。

全局口径：三个任务合计 **445 episodes / 379,564 rows / 137 个实际唯一 bare-English low-level instruction**；其中 `Clean The Sink` 45 个、`Clean The Cutting Board` 57 个、`Wipe The Desktop` 35 个。

</details>
