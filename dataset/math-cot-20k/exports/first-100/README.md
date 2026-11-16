# Math-CoT-20k：原始顺序前 100 条

[返回数据集说明](../../README.md) · [原始 Parquet](../../raw/Math-CoT-20k.parquet) · [人工标注表](annotations.csv) · [选取记录](selection.json) · [另看最长 100 条](../longest-100/README.md)

按原始 Parquet 文件顺序取前 100 条，即从 0 开始的行号 0—99。没有按长度排序、随机抽样或去重，也不是之前正式训练的 400 条。

原始数据共有 20,480 条；本次导出 100 条；数据集记录的解答长度范围为 1,843—16,381。

## 怎么阅读和挑选

1. 点击目录里的序号进入单题；每页包含上一条、下一条和返回目录。
2. 每题按题目、参考答案、完整回答排列，完整回答放在最后。全文原样放入文本块，保留推理标签及换行；长行可在编辑器中开启自动换行。
3. 在 annotations.csv 中填写 status（选中／排除／待定）、tags 和 notes，保留原始行号和 sample_id。CSV 使用 UTF-8 BOM。
4. 页面笔记与 CSV 不自动同步，统一清单以 CSV 为准；已有最长 100 条的文件和标注保持独立。

## 训练资格

本导出没有进行聊天模板编码、10,240 token 长度过滤、holdout/评测重合检查或参考答案核验。response_length 仅是发布者记录的长度。后续用于正式训练时，仍需经过项目的数据准备流程。

## 阅读目录

| 序号 | 原始行号（0 起） | 记录长度 | 来源 | 题目摘要（仅目录缩略） |
| --- | ---: | ---: | --- | --- |
| [001](samples/001-row-00000.md) | 0 | 7,580 | olympiads | In $\triangle A B C$, the three sides $a, b, c$ form an arithmetic sequence. Find the value of $5 \c… |
| [002](samples/002-row-00001.md) | 1 | 7,230 | olympiads | Example 13 In the unit cube $A B C D-A_{1} B_{1} C_{1} D_{1}$, $E$ and $F$ are the midpoints of $A B… |
| [003](samples/003-row-00002.md) | 2 | 5,511 | aops_forum | Construct a geometric gure in a sequence of steps. In step $1$, begin with a $4\times 4$ square. In … |
| [004](samples/004-row-00003.md) | 3 | 7,060 | cn_contest | 1. Let $p, q \in N$, and $1 \leqslant p&lt;q \leqslant n$, where $n$ is a natural number not less than … |
| [005](samples/005-row-00004.md) | 4 | 6,865 | cn_contest | 4. As shown in Figure 1, it is known that $A B C D-A_{1} B_{1} C_{1} D_{1}$ is a cube with a side le… |
| [006](samples/006-row-00005.md) | 5 | 11,384 | olympiads | 7. In tetrahedron $A B C D$, $\angle A D B=\angle B D C$ $=\angle C D A=60^{\circ}, A D=B D=3, C D=2… |
| [007](samples/007-row-00006.md) | 6 | 6,243 | cn_contest | 4. For all real numbers $x$, the values of all quadratic functions $f(x) = ax^2 + bx + c (a \leqslan… |
| [008](samples/008-row-00007.md) | 7 | 14,134 | aops_forum | Find the number of pairs $(m,n)$ of integers with $-2014\le m,n\le 2014$ such that $x^3+y^3 = m + 3n… |
| [009](samples/009-row-00008.md) | 8 | 3,368 | olympiads | ## Problem Statement Calculate the limit of the function: $\lim _{x \rightarrow 1}\left(\frac{e^{2 x… |
| [010](samples/010-row-00009.md) | 9 | 3,922 | olympiads | 3. Find the number of natural numbers $k$, not exceeding 267000, such that $k^{2}-1$ is divisible by… |
| [011](samples/011-row-00010.md) | 10 | 5,336 | olympiads | One. (This question is worth 25 points) Let the sequence $\left\{a_{n}\right\}$ have the sum of the … |
| [012](samples/012-row-00011.md) | 11 | 7,161 | olympiads | 8. Let $g(x)=\sqrt{x(1-x)}$ be a function defined on the interval $&#91;0,1&#93;$, then the area of the figu… |
| [013](samples/013-row-00012.md) | 12 | 6,925 | olympiads | ## Problem Statement Calculate the definite integral: $$ \int_{0}^{2 \pi} \sin ^{6}\left(\frac{x}{4}… |
| [014](samples/014-row-00013.md) | 13 | 4,945 | cn_contest | Example 2. 8 different elements are arranged in two rows, with 4 elements in each row, where 2 speci… |
| [015](samples/015-row-00014.md) | 14 | 10,217 | olympiads | XLVII OM - III - Problem 1 Determine all pairs $ (n,r) $, where $ n $ is a positive integer and $ r … |
| [016](samples/016-row-00015.md) | 15 | 4,154 | olympiads | Task 1. Determine a prime number $p$, such that the number $2 p+1$ is a perfect cube of some natural… |
| [017](samples/017-row-00016.md) | 16 | 8,391 | olympiads | A Pretti number is a seven-digit positive integer with the following properties: - The integer forme… |
| [018](samples/018-row-00017.md) | 17 | 6,315 | olympiads | 16. Among the following four numbers, which can be written as the sum of 100 consecutive natural num… |
| [019](samples/019-row-00018.md) | 18 | 4,484 | cn_contest | 1. Given the sets $M=\{x, x y, \lg (x y)\}$ and $N=\{0,&#124;x&#124;, y\}$, and $M=N$. Then, $\left(x+\frac{1}… |
| [020](samples/020-row-00019.md) | 19 | 6,994 | olympiads | 1. Given a linear function $f(x)$. It is known that the distance between the points of intersection … |
| [021](samples/021-row-00020.md) | 20 | 7,065 | cn_contest | 5. As shown in Figure 2, in square $ABCD$, $E$ and $F$ are points on sides $AB$ and $BC$ respectivel… |
| [022](samples/022-row-00021.md) | 21 | 5,524 | olympiads | 10,11 &#124; &#124; On the lateral edge of the pyramid, two points are taken, dividing the edge into three equ… |
| [023](samples/023-row-00022.md) | 22 | 6,681 | aops_forum | Fix positive integers $n,j,k$.How many integer sequences are there of the form $1\le a_1&lt;a_2&lt;\ldots&lt;… |
| [024](samples/024-row-00023.md) | 23 | 3,957 | olympiads | 7. There is a rectangular iron sheet of size $80 \times 50$. Now, a square of the same size is to be… |
| [025](samples/025-row-00024.md) | 24 | 5,811 | olympiads | Problem 8.5. A car left the city for the village at the same time a cyclist left the village for the… |
| [026](samples/026-row-00025.md) | 25 | 3,580 | olympiads | Let's determine the prime number $p$, knowing that it has an even number of digits, and the number i… |
| [027](samples/027-row-00026.md) | 26 | 3,029 | olympiads | 6. (5 points) Using the digits $2$, $0$, $1$, and $5$, we can form $\qquad$ unique three-digit numbe… |
| [028](samples/028-row-00027.md) | 27 | 6,099 | inequalities | $4 \cdot 64$ Find the product of the real roots of the equation $x^{2}+18 x+30=2 \sqrt{x^{2}+18 x+45… |
| [029](samples/029-row-00028.md) | 28 | 15,623 | cn_contest | Example 2. Given that $a, b$ are two unequal positive numbers, among the following three algebraic e… |
| [030](samples/030-row-00029.md) | 29 | 3,560 | olympiads | 7.293. $\left\{\begin{array}{l}(x+y)^{x}=(x-y)^{y}, \\ \log _{2} x-\log _{2} y=1\end{array}\right.$ … |
| [031](samples/031-row-00030.md) | 30 | 3,961 | olympiads | Aerith bakes some cookes. On the first day, she gives away 1 cookie and then $1 / 8$ of the remainin… |
| [032](samples/032-row-00031.md) | 31 | 10,583 | olympiads | Example 4 As shown in Figure 4, in $\triangle A B C$, $A B=A C, \angle A$ $=100^{\circ}, I$ is the i… |
| [033](samples/033-row-00032.md) | 32 | 2,660 | olympiads | I4.1 Consider the function $y=\sin x+\sqrt{3} \cos x$. Let $a$ be the maximum value of $y$. Find the… |
| [034](samples/034-row-00033.md) | 33 | 9,728 | cn_contest | 6. Four people, A, B, C, and D, are playing a passing game. In the first pass, A passes the ball to … |
| [035](samples/035-row-00034.md) | 34 | 8,095 | aops_forum | Let $\mathcal{P}_1$ and $\mathcal{P}_2$ be two parabolas with distinct directrices $\ell_1$ and $\el… |
| [036](samples/036-row-00035.md) | 35 | 4,439 | olympiads | 4. If the positive integer $n$ makes $\sqrt{3}$ always lie between $1+\frac{3}{n}$ and $1+\frac{3}{n… |
| [037](samples/037-row-00036.md) | 36 | 4,371 | olympiads | 6. Two people take turns rolling dice, each rolling two at a time. The first person to roll a sum gr… |
| [038](samples/038-row-00037.md) | 37 | 3,421 | olympiads | 9. In the example $(* *)^{3}=* * * 9$, replace the asterisks with digits to obtain a correct equatio… |
| [039](samples/039-row-00038.md) | 38 | 11,139 | cn_contest | 9. Let the positive integer $n$ satisfy $31 \mid\left(5^{n}+n\right)$. Then the minimum value of $n$… |
| [040](samples/040-row-00039.md) | 39 | 4,033 | olympiads | 1. If the complex number $z$ satisfies $&#124;z-1&#124;+&#124;z-3-2 \mathrm{i}&#124;=2 \sqrt{2}$, then the minimum value… |
| [041](samples/041-row-00040.md) | 40 | 6,260 | cn_contest | 1. Factorize $(b-c)^{3}+(c-a)^{3}+(a-b)^{3}$ |
| [042](samples/042-row-00041.md) | 41 | 6,797 | olympiads | 20th APMC 1997 Problem 3 The 97 numbers 49/1, 49/2, 49/3, ... , 49/97 are written on a blackboard. W… |
| [043](samples/043-row-00042.md) | 42 | 7,757 | olympiads | 2. The coefficient of $x^{150}$ in the expansion of $\left(1+x+x^{2}+\cdots+x^{100}\right)^{3}$, aft… |
| [044](samples/044-row-00043.md) | 43 | 5,618 | olympiads | G5.2 Let $b$ be the maximum of the function $y=\left&#124;x^{2}-4\right&#124;-6 x$ (where $-2 \leq x \leq 5$ )… |
| [045](samples/045-row-00044.md) | 44 | 5,037 | cn_contest | 3. Given planar vectors $\boldsymbol{a}=(\sqrt{3},-1), \boldsymbol{b}=$ $\left(\frac{1}{2}, \frac{\s… |
| [046](samples/046-row-00045.md) | 45 | 7,071 | olympiads | 4. Given the sequence $a_{1}=1, a_{n}=\frac{n}{n^{2}-1}(n \geqslant 2)$, then $\sum_{k=1}^{n}\left(a… |
| [047](samples/047-row-00046.md) | 46 | 2,322 | olympiads | How many of the four integers $222,2222,22222$, and 222222 are multiples of 3 ? (A) 0 (B) 1 (C) 2 (D… |
| [048](samples/048-row-00047.md) | 47 | 3,601 | olympiads | 14. Two students, A and B, who live in the same community, leave the community gate for school at th… |
| [049](samples/049-row-00048.md) | 48 | 14,318 | olympiads | 28. (ITA 3) Find values of the parameter $u$ for which the expression $$ y=\frac{\tan (x-u)+\tan x+\… |
| [050](samples/050-row-00049.md) | 49 | 5,459 | olympiads | 8.2. (England, 75). Solve the equation $$ &#91;\sqrt&#91;3&#93;{1}&#93;+&#91;\sqrt&#91;3&#93;{2}&#93;+\ldots+\left&#91;\sqrt&#91;3&#93;{x^{3}-1}… |
| [051](samples/051-row-00050.md) | 50 | 13,802 | olympiads | 3. As shown in Figure 7, points $A$ and $C$ are on ray $O M$, and points $B$ and $D$ are on ray $O N… |
| [052](samples/052-row-00051.md) | 51 | 2,926 | olympiads | For example, in the sequence $x_{1}, x_{2}, \cdots, x_{n}, \cdots$, the sum of any three consecutive… |
| [053](samples/053-row-00052.md) | 52 | 1,843 | olympiads | ## Task B-1.1. The sum of two numbers is 6. If the sum of their cubes is 90, what is the sum of thei… |
| [054](samples/054-row-00053.md) | 53 | 12,493 | olympiads | 4. Find the smallest value of the parameter $a$, for which the system of equations $$ \left\{\begin{… |
| [055](samples/055-row-00054.md) | 54 | 5,311 | olympiads | 3. Triangle $A B C$ is equilateral. On side $A C$, point $M$ is marked, and on side $B C$, point $N$… |
| [056](samples/056-row-00055.md) | 55 | 7,411 | cn_contest | Example 2 If $f(x) (x \in \mathbf{R})$ is an even function with a period of 2, and when $x \in &#91;0,1&#93;… |
| [057](samples/057-row-00056.md) | 56 | 3,994 | olympiads | 2. Let $a&gt;0$, if the solution set of the inequality $\sqrt{x+a}+\sqrt{x-a} \leqslant \sqrt{2(x+1)}$ … |
| [058](samples/058-row-00057.md) | 57 | 4,284 | cn_contest | 6. Given a cube $A B C D-A_{1} B_{1} C_{1} D_{1}$ with edge length $1, O$ is the center of the base … |
| [059](samples/059-row-00058.md) | 58 | 4,785 | olympiads | 4. Find the minimum value of the sum $$ \left&#124;x-1^{2}\right&#124;+\left&#124;x-2^{2}\right&#124;+\left&#124;x-3^{2}\righ… |
| [060](samples/060-row-00059.md) | 59 | 5,377 | olympiads | 9. Given the sequence $\left\{a_{n}\right\}$ satisfies $a_{1}=1, a_{n+1} a_{n-1}=a_{n} a_{n-1}+a_{n}… |
| [061](samples/061-row-00060.md) | 60 | 4,526 | olympiads | 15. Random Vectors. There are $n$ random vectors of the form $\left(y_{1}, y_{2}, y_{3}\right)$, whe… |
| [062](samples/062-row-00061.md) | 61 | 5,020 | olympiads | 324. $y=\frac{1}{2} x^{3}, x=-2, x=4$ and $y=0$. |
| [063](samples/063-row-00062.md) | 62 | 6,497 | olympiads | 7. Let $\triangle A B C$ be a right triangle with right angle $C$. Let $I$ be the incenter of $A B C… |
| [064](samples/064-row-00063.md) | 63 | 3,520 | olympiads | 3. Among the four functions $y=\sin &#124;x&#124;, y=\cos &#124;x&#124;, y=&#124;\cot x&#124;, y=\lg &#124;\sin x&#124;$, the even function … |
| [065](samples/065-row-00064.md) | 64 | 5,375 | olympiads | 15. (15 points) 4 consecutive natural numbers, from smallest to largest, are multiples of 11, 7, 5, … |
| [066](samples/066-row-00065.md) | 65 | 2,888 | olympiads | 10.090. The distance from the center of the circle to a chord of length 16 cm is 15 cm. Find the are… |
| [067](samples/067-row-00066.md) | 66 | 9,948 | cn_contest | 10. Among three-digit numbers, if the digit in the tens place is smaller than the digits in the hund… |
| [068](samples/068-row-00067.md) | 67 | 2,341 | olympiads | 910. A stationary random function $X(t)$ with mean $m_{x}=5$ is input to a linear stationary dynamic… |
| [069](samples/069-row-00068.md) | 68 | 6,952 | olympiads | 1. From Central Square to the station, there is a straight street divided by 11 intersections into 1… |
| [070](samples/070-row-00069.md) | 69 | 16,381 | cn_contest | 3. A book has a total of 61 pages, sequentially numbered as 1, 2, ..., 61. Someone, while adding the… |
| [071](samples/071-row-00070.md) | 70 | 8,869 | olympiads | 2. Point $B$ is internal to segment $\overline{A C}$. In the same half-plane relative to line $A C$,… |
| [072](samples/072-row-00071.md) | 71 | 4,123 | cn_contest | 3. Let $a$, $b$, and $c$ be positive integers, and $a \geqslant b \geqslant c$, satisfying $a+b+c=15… |
| [073](samples/073-row-00072.md) | 72 | 4,472 | olympiads | Let $f: \mathbb{R} \rightarrow \mathbb{R}$ be an odd function, that is, a function that satisfies $-… |
| [074](samples/074-row-00073.md) | 73 | 7,862 | olympiads | Problem 11.6. Given a convex quadrilateral $A B C D$. It is known that $\angle C=57^{\circ}, \sin \a… |
| [075](samples/075-row-00074.md) | 74 | 5,815 | cn_contest | 2. If the graph of the function $f(x)=a^{2} \sin 2 x+(a-2) \cos 2 x$ is symmetric about the line $x=… |
| [076](samples/076-row-00075.md) | 75 | 13,322 | olympiads | $13 \cdot 50$ The function $f(x)=\sin x+\sin 2 x+\sin 3 x$ has (A) a maximum value of 3. (B) a maxim… |
| [077](samples/077-row-00076.md) | 76 | 8,633 | olympiads | 6. On the plane, there are two fixed points $A$ and $B$, and 4 moving points $C_{1}$, $C_{2}$, $C_{3… |
| [078](samples/078-row-00077.md) | 77 | 3,713 | amc_aime | Assume that $x_1,x_2,\ldots,x_7$ are real numbers such that \begin{align*} x_1 + 4x_2 + 9x_3 + 16x_4… |
| [079](samples/079-row-00078.md) | 78 | 8,101 | olympiads | 3. Find all prime numbers of the form 1010101...0101 (i.e., whose decimal representation consists of… |
| [080](samples/080-row-00079.md) | 79 | 5,149 | olympiads | 1. First solution: Since the Rabbit ran at a speed twice that of Alice, by the time Alice arrived at… |
| [081](samples/081-row-00080.md) | 80 | 3,843 | olympiads | 7. A certain two-digit number is a multiple of 2, adding 1 makes it a multiple of 3, adding 2 makes … |
| [082](samples/082-row-00081.md) | 81 | 2,765 | olympiads | 5. (7 points) Four girls are singing songs, accompanying each other. Each time, one of them plays th… |
| [083](samples/083-row-00082.md) | 82 | 2,168 | olympiads | 2. For the sequence $u_{1}, u_{2}, \cdots$, define $\Delta^{\prime}\left(u_{n}\right)=u_{n+1}-u_{n}$… |
| [084](samples/084-row-00083.md) | 83 | 5,982 | olympiads | # Task № 4.2 ## Condition: Six friends live in a small town in southern France. One day they decided… |
| [085](samples/085-row-00084.md) | 84 | 11,793 | olympiads | 1. Let the side length of the base of a regular quadrilateral pyramid be 2017, and the side edge len… |
| [086](samples/086-row-00085.md) | 85 | 2,051 | amc_aime | The base of a new rectangle equals the sum of the diagonal and the greater side of a given rectangle… |
| [087](samples/087-row-00086.md) | 86 | 2,992 | olympiads | If a class of 30 people is seated in a movie theater, then in any case, at least two classmates will… |
| [088](samples/088-row-00087.md) | 87 | 2,746 | cn_contest | 3. If the focal distance, minor axis length, and major axis length of an ellipse form an arithmetic … |
| [089](samples/089-row-00088.md) | 88 | 10,204 | aops_forum | There are $n$ students standing in line positions $1$ to $n$. While the teacher looks away, some stu… |
| [090](samples/090-row-00089.md) | 89 | 9,773 | olympiads | 8. In tetrahedron $ABCD$, the dihedral angle between faces $ABC$ and $BCD$ is $30^{\circ}$, the area… |
| [091](samples/091-row-00090.md) | 90 | 3,056 | olympiads | 6.139. $(x+1)^{5}+(x-1)^{5}=32 x$. Translate the above text into English, please retain the original… |
| [092](samples/092-row-00091.md) | 91 | 12,328 | olympiads | 14. New Year's Problem (recommended for 8th grade, 4 points). On the New Year's table, there are 4 g… |
| [093](samples/093-row-00092.md) | 92 | 8,441 | olympiads | ## Task A-3.1. (4 points) Solve the equation $\quad \sin x \cdot \cos 2 x \cdot \cos 4 x=1$. |
| [094](samples/094-row-00093.md) | 93 | 2,431 | olympiads | 9. (3 points) At Dongfang Primary School, Grade 6 selected $\frac{1}{11}$ of the girls and 22 boys t… |
| [095](samples/095-row-00094.md) | 94 | 6,215 | olympiads | 5. In the coordinate plane, a point whose both coordinates are integers is called an integer point. … |
| [096](samples/096-row-00095.md) | 95 | 5,298 | olympiads | ## Problem Statement Calculate the volumes of the bodies bounded by the surfaces. $$ \frac{x^{2}}{27… |
| [097](samples/097-row-00096.md) | 96 | 2,238 | olympiads | 3B. Let $z$ and $w$ be complex numbers such that $&#124;z&#124;=&#124;w&#124;=&#124;z-w&#124;$. Calculate the value of $\left(\fra… |
| [098](samples/098-row-00097.md) | 97 | 5,024 | amc_aime | Let $f(x)$ be a &#91;polynomial&#93;(https://artofproblemsolving.com/wiki/index.php/Polynomial) with real &#91;c… |
| [099](samples/099-row-00098.md) | 98 | 3,127 | olympiads | IS. 1 In figure $1, \triangle A B C$ is an acute triangle, $A B=15, A C=13$, and its altitude $A D=1… |
| [100](samples/100-row-00099.md) | 99 | 7,659 | cn_contest | 14. B. Color the five sides and five diagonals of the convex pentagon $A B C D E$, such that any two… |
