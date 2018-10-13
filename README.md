# TicTacToe

## 原理

主要是看了这一篇[Get a taste of reinforcement learning — implement a tic tac toe agent](https://medium.com/@shiyan/get-a-taste-of-reinforcement-learning-implement-a-tic-tac-toe-agent-deda5617b2e4)，里面作者提出了大概的训练思路，我基本没有参照他的具体实现，但是思路肯定是差不多的。而且训练结果是我几乎下不过这个AI，比这篇的结果要好很多（不过这种明显算法可以解决的问题用AI也没什么意思）。

## 实现
核心在于训练一个模型$M$，这个模型用来学习当前的棋局$G$时，我们的当前玩家$p$该怎么处理。强化学习不需要训练用例，而是我们自己产生训练用例。这是一个有点循环依赖的问题。因为模型显然自己不知道该往什么方向发展，通常的机器学习训练中，我们的依靠外在的数据让模型去调整自己的参数。所幸，我们的$G$服从一套游戏规则，这个游戏规则可以帮助我们产生（越来越好的）训练用例。

#### 训练过程

我们的训练流程是这样的。圈圈用符号$O$，叉叉用符号$X$表示。先手圈圈的模型称为$M_O$，后手叉叉的模型称为$M_X$，他们需要学习一个映射，映射到当前奖励$E_p$。$E_p$代表模型对当前各种走法（圈叉棋中至多9种）的价值的判断。即
$$M_O = G\mapsto E_O$$
$$M_X = G\mapsto E_X$$
1. 初始状态时我们随机初始化$M_O$ 和 $M_X$。
2. 然后我们利用当前参数的$M_O$，$M_X$在盘面上进行搏斗，即产生（一个batch的）训练用例。为了产生更广泛的训练样例，我们采用算法$A$来产生随机的训练用例(之后定义算法$A$)。
3. 利用训练用例调整$M_O$和$M_X$的参数，返回 2。

#### 算法$A$

1. 随机产生 $b$ 盘合法棋局$S_g$，这些棋局满足游戏规则，而且当前不存在胜者。
2.  对于每个棋局 $s\in S_g$ ，如果是轮到选手 $O$， 则认为$s \in S_O$，否则 $s \in S_X$。
3. 对于每个棋局 $s\in S_g$ ，按照游戏规则产生用例。对当前$s$中每个空格，填入当前选手的符号。之后按照当前选手，通过$M_O$ 或 $M_X$预测奖励，选择奖励最大的位置进行走子（我们会解决当前模型预测奖励最大的位置不是空格的问题*）。直到当前棋局 $s$ 分出胜负（或平局）。对每个空格处理完后，我们可以对当前棋局$s$按照如下规则进行估算奖励$E_{p,s}$，注意这里的$p$是棋局$s$的当前玩家，$E_{p,s}$可以当作一个9元向量，$E_{p,s}^i$是上文所说的当前走位置$i$的奖励，假设我们把棋局$s$编号到$k\in[0,9)$的整数。：
	* 如果 $s^i$ 已经被占了，那么 $E_{p,s}^i$ 定义为 0。
	* 如果 $s^i$ 在上述过程第一步"填入当前选手的符号"后，经过 $step$ 步胜利了，定义 $E_{p,s}^i = V(step)$，这里的 $V$ 是我们自己定义的函数，它可以是常函数 `lambda V : 1` 也可以是某个关于 `step` 的减函数，来促使我们的模型尽早胜利。我们可以定义最优的 $E_{p,s}^i = 1$。
	* 如果 2 中，经过 $step$ 步后平局或者没有胜利，定义 $E_{p,s}^i = F(step)$。这个 F 也可以自己来定义。根据我的经验，F是$step$的增函数会促使模型学会堵子。
4. 如此一来我们就有了分别面向模型$M_O$和$M_X$的训练集$T_O$和$T_X$。
$$T_O = \{ (s, E_{O, s})|s\in S_O \}$$
$$T_X = \{ (s, E_{X, s})|s\in S_X \}$$

#### 理解

模型训练的核心显然在我们为什么用算法$A$能产生对$M_O$和$M_X$有优化作用的训练用例集$T_O$和$T_X$。
正如训练过程(1)所示，一开始$M_O$和$M_X$的参数是随机的，所以我们左右互搏产生的测试用例实际是质量很低的，因为双方并没有建立起如实反映$G\mapsto E_P$的映射$M_P$。但是我们有强制的一步算法$A$的(3)，我们对每个空格都进行了试验，因此至少获得了一部分 $S'\subset S_g$ 的真实 $E_{p,s}$ 值，例如
$$
s  = 
  \begin{matrix}
   O & O & \_ \\
   X & X & \_ \\
   \_ & \_ & \_
  \end{matrix}
  \in S'
$$

当前选手是$O$，那么。之后我们对$E_{o,s}^2 = V(1)$ (假设位置编号从左到右从上到下，从0开始。)便得到了可靠的目标值。然后我们在训练过程 3. 中利用优化算法，便使得我们的模型对 $S'$ 有了更加准确的映射。如此而来，由于 $M_p$ 变得更加准确了，下一轮便能得到更加高质量的训练用例。例如当
$$
s'  = 
  \begin{matrix}
   O & O & \_ \\
   X & \_ & \_ \\
   \_ & \_ & \_
  \end{matrix}
  \in S'
$$
轮到$X$走子，对于试验填入位置 4 ，那么由于我们的模型$M_O$对 $E_{O,s}^2$ 有较高的价值判断（从而导致此次试验$X$输），因此$X$将学习避免走位置 $4$。这样我们的模型就学习到了$S'$以外的正确映射。
当然，这是非常不mathematic的，但是可以作为一个肤浅的理解。

## 实验

1. 代码 https://github.com/hanayashiki/TicTacToe  (python 3.6, keras, tensorflow)。
2. 棋局用 $G \in \mathbb{R^{3\times3\times3}}$ 表示，其中最后一个维度是 $(1, 0, 0)$ 表示填入 ”O"，$(0, 1, 0)$ 表示是空白，$(0, 0, 1)$ 表示填入 "X"，采用独热编码是为了尽量减少假设。
3. 表现最好的模型定义在 `model.py` 中的 `ModelThreeDensesReluReluAdamMasked`。它的结构是
	1. `shape=(27,)` 的输入层
	2.  两层输出为 256 维的全连接层，激活函数为 `relu`
	3.  全连接输出层为 9 维，激活函数为 `relu`
	4. mask 层生成 9 维 0-1 mask 对应当前能走的点位 *，因此不能走的地方总是 0，我们的算法不需要学习不符合规则的情况。mask 层按元素与 3 层相连接。 同时由于 `relu` 可能有全 0 的输出，我们接着在 mask 层对应的位置加上一个小量 0.0001，避免最终预测结果全 0。
	5. 算法 $A$ 中的 $V$ 和 $F$ 函数进行瞎几把定义，我这里保证平局或者输产生的 $E_{s,p}$ 取值在 $[0, 0.4]$，随步数增加；胜利产生的 $E_{s,p}$ 在 $[0.7, 1]$ 之间。直觉地促使如果输，那么要尽量拖延（学会堵子）；如果赢，尽量用较少的步数（学会将死）。
4. 训练时每次产生 64 个测试用例。loss function 使用 `MSE`。

## 结果
到了见证结果的激动人心的时间！
经过 3000 轮训练，$M_O$的`loss` 达到了 `0.002995`，$M_X$的`loss` 达到了 `0.00329`，这说明机器左右互搏的预测已经相当准确了，那么和人下一盘呢？
```
New game!
>> Select your role: 'X' or 'O'. o

_ _ _ 
_ _ _ 
_ _ _ 


key >>s // 下方是我的走法

_ _ _ 
_ O _ 
_ _ _ 



_ _ X 
_ O _ 
_ _ _ 


key >>c

_ _ X 
_ O _ 
_ _ O 



X _ X 
_ O _ 
_ _ O 


key >>w

X O X 
_ O _ 
_ _ O 



X O X 
_ O _ 
_ X O 


key >>a

X O X 
O O _ 
_ X O 



X O X 
O O X 
_ X O 


key >>z

X O X 
O O X 
O X O 


It's a tie !
```
可见电脑后手有一定的智能，和我打平了。它的先手更有出乎意料的操作。

```
>> Select your role: 'X' or 'O'. X

_ _ _ 
_ _ _ 
_ _ _ 



_ _ O // 电脑没有像我们人一样选择中间位置
_ _ _ 
_ _ _ 


key >>s

_ _ O 
_ X _ 
_ _ _ 



_ _ O 
_ X _ 
O _ _ 


key >>c

_ _ O 
_ X _ 
O _ X 



O _ O 
_ X _  // 将死了随便下的我
O _ X 


key >>x

O _ O 
_ X _ 
O X X 



O O O 
_ X _ 
O X X 


Player O wins ! 
```
极限操作方可平局：
```
New game!
>> Select your role: 'X' or 'O'. x

_ _ _ 
_ _ _ 
_ _ _ 



_ _ O 
_ _ _ 
_ _ _ 


key >>s

_ _ O 
_ X _ 
_ _ _ 



_ _ O 
_ X _ 
O _ _ 


key >>a

_ _ O 
X X _ 
O _ _ 



_ _ O 
X X O 
O _ _ 


key >>c

_ _ O 
X X O 
O _ X 



O _ O 
X X O 
O _ X 


key >>w

O X O 
X X O 
O _ X 



O X O 
X X O 
O O X 


It's a tie !

```
看看机器左右互搏的情况(最后是平局)
```
_ _ O 
_ _ _ 
_ _ _ 

_ _ O 
_ X _ 
_ _ _ 

_ _ O 
_ X _ 
O _ _ 

_ _ O 
_ X _ 
O X _ 

_ O O 
_ X _ 
O X _ 

X O O 
_ X _ 
O X _ 

X O O 
_ X _ 
O X O 

X O O 
_ X X 
O X O 

X O O 
O X X 
O X O 
```

总之这个训练结果，非常有意思！机器仅仅从规则学习到的操作，出乎人类玩家意料！
