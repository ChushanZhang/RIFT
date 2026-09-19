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

完整的 **137 种训练文本、中文和出现次数**见[原文与统计附录](galaxea_training_instruction_inventory.md)。其中 `null` 是异常标注，不作为操作指令；“本体未动”一类文本也不作为流程步骤或机器人暂停按钮。

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
