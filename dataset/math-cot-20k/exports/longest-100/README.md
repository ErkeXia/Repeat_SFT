# Math-CoT-20k：解答最长的 100 条

[返回数据集说明](../../README.md) · [原始 Parquet](../../raw/Math-CoT-20k.parquet) · [人工标注表](annotations.csv) · [选取记录](selection.json)

这里的“前 100 条”指按 response_length 从长到短排序后的前 100 条，并非原文件最前面的 100 行。相同长度按原始行号升序排列。

原始数据共有 20,480 条；本次选取 100 条；记录长度范围 15,643—16,381。

## 怎么阅读和挑选

1. 点击下面的序号，进入单条 Markdown 文件；每页有上一条、下一条和返回目录。
2. 每页按题目、参考答案、完整回答排列。完整回答位于最后，保留全部原文，用文本块展示以免推理标签或 Markdown 符号被隐藏。长行可在编辑器中开启自动换行。
3. 在 annotations.csv 中填写 status（选中／排除／待定）、tags 和 notes。保留 rank、row_index 和 sample_id，方便回到原记录。CSV 使用 UTF-8 BOM，便于表格软件打开。
4. 页面上的笔记区与 CSV 不自动同步；统一挑选清单以 CSV 为准。完整原始数据始终保留在 raw 目录。

## 长度和训练限制

response_length 是发布者记录的解答长度，不是本项目套用聊天模板后重新计算的总 token 数。当前训练上限是 10,240 token；本批是原始数据中超长样本的阅读材料，尚未进行训练资格筛选，也不代表此前实际训练的 400 道题。若用于当前配置的训练，应从服务器准备好的 pool.jsonl 中按 n_supervised 挑选，并保留完整核验流程。

## 阅读目录

| 序号 | 原始行号（0 起） | 记录长度 | 来源 | 题目摘要（仅目录缩略） |
| --- | ---: | ---: | --- | --- |
| [001](samples/001-row-00069.md) | 69 | 16,381 | cn_contest | 3. A book has a total of 61 pages, sequentially numbered as 1, 2, ..., 61. Someone, while adding the… |
| [002](samples/002-row-00694.md) | 694 | 16,376 | amc_aime | Let $S$ be the sum of all numbers of the form $a/b,$ where $a$ and $b$ are &#91;relatively prime&#93;(https:… |
| [003](samples/003-row-06621.md) | 6621 | 16,366 | olympiads | ## SUBJECT I Determine the non-zero natural number n such that $\frac{2}{3 \cdot 5}+\frac{2}{5 \cdot… |
| [004](samples/004-row-07666.md) | 7666 | 16,362 | aops_forum | Find all sets $X$ consisting of at least two positive integers such that for every two elements $m,n… |
| [005](samples/005-row-01472.md) | 1472 | 16,340 | cn_contest | 10. In tetrahedron $ABCD$, $AB=CD=6$, $AC=$ $AD=BC=BD=5$. Then the radius $r$ of the inscribed spher… |
| [006](samples/006-row-11323.md) | 11323 | 16,321 | olympiads | 2. In the given parallelogram $A B C D$, point $E$ is the midpoint of side $B C$ and point $F$ lies … |
| [007](samples/007-row-05225.md) | 5225 | 16,316 | olympiads | 8. The school stage of the Magic and Wizardry Olympiad consists of 5 spells. Out of 100 young wizard… |
| [008](samples/008-row-19678.md) | 19678 | 16,316 | olympiads_ref | Problem. Find all functions $f:(0, \infty) \rightarrow(0, \infty)$ such that $$ f\left(y(f(x))^{3}+x… |
| [009](samples/009-row-13368.md) | 13368 | 16,315 | aops_forum | Compute the largest integer $N$ such that one can select $N$ different positive integers, none of wh… |
| [010](samples/010-row-10305.md) | 10305 | 16,305 | cn_contest | Example 7 For a natural number $k, g(k)$ represents the greatest odd divisor of $k$ (for example, $g… |
| [011](samples/011-row-15384.md) | 15384 | 16,277 | aops_forum | Find all prime numbers $p$ and $q$ such that $$1 + \frac{p^q - q^p}{p + q}$$ is a prime number. &#91;i&#93;P… |
| [012](samples/012-row-01063.md) | 1063 | 16,271 | cn_contest | 4. As shown in Figure 3, in $\triangle A B C$, $M$ is the midpoint of side $B C$, and $M D \perp A B… |
| [013](samples/013-row-07511.md) | 7511 | 16,267 | olympiads | 5. In the spatial quadrilateral $ABCD$, it is known that $AB=2, BC=3, CD=4, DA=5$, then $\overrighta… |
| [014](samples/014-row-14552.md) | 14552 | 16,264 | olympiads_ref | Determine all pairs $(a, b)$ of positive integers for which $$ a^{2}+b \mid a^{2} b+a \quad \text { … |
| [015](samples/015-row-12915.md) | 12915 | 16,257 | olympiads_ref | Let $\mathbb{R}_{&gt;0}$ be the set of positive real numbers. Find all functions $f: \mathbb{R}_{&gt;0} \r… |
| [016](samples/016-row-17705.md) | 17705 | 16,256 | aops_forum | Let $n$ be a positive integer such that $\lfloor\sqrt n\rfloor-2$ divides $n-4$ and $\lfloor\sqrt n\… |
| [017](samples/017-row-06642.md) | 6642 | 16,242 | aops_forum | The polynomial of seven variables $$ Q(x_1,x_2,\ldots,x_7)=(x_1+x_2+\ldots+x_7)^2+2(x_1^2+x_2^2+\ldo… |
| [018](samples/018-row-20084.md) | 20084 | 16,236 | olympiads | ## Problem Statement Calculate the limit of the function: $\lim _{x \rightarrow \frac{\pi}{2}} \frac… |
| [019](samples/019-row-10008.md) | 10008 | 16,229 | olympiads | 11. (20 points) Let the sequences $\left\{a_{n}\right\}$ and $\left\{b_{n}\right\}$ satisfy: $$ a_{1… |
| [020](samples/020-row-19061.md) | 19061 | 16,228 | cn_contest | 3. A large cube is formed by $8 \times 8 \times 8$ unit cubes. How many lines can pass through the c… |
| [021](samples/021-row-18210.md) | 18210 | 16,198 | olympiads | 5. Several people played a round-robin table tennis tournament. At the end of the tournament, it tur… |
| [022](samples/022-row-18750.md) | 18750 | 16,193 | cn_contest | One. (35 points) The lines $l_{1}$ and $l_{2}$ are tangent to a circle at points $A$ and $B$, respec… |
| [023](samples/023-row-14534.md) | 14534 | 16,190 | olympiads | King Arthur's bodyguards held a knightly tournament. At the end of the tournament, it turned out tha… |
| [024](samples/024-row-10470.md) | 10470 | 16,183 | olympiads | 2. The left and right foci of a hyperbola are $F_{1}, F_{2}$, respectively. A line passing through $… |
| [025](samples/025-row-10998.md) | 10998 | 16,157 | aops_forum | Let $I$ be the set of points $(x,y)$ in the Cartesian plane such that $$x&gt;\left(\frac{y^4}{9}+2015\r… |
| [026](samples/026-row-05392.md) | 5392 | 16,144 | olympiads | 9. Let $m$ be a positive integer, and let $T$ denote the set of all subsets of $\{1,2, \ldots, m\}$.… |
| [027](samples/027-row-18795.md) | 18795 | 16,129 | aops_forum | If five squares of a $3 \times 3$ board initially colored white are chosen at random and blackened, … |
| [028](samples/028-row-20298.md) | 20298 | 16,125 | cn_contest | 6. Let $0 \leqslant x \leqslant 8$. Then $$ f(x)=\frac{\sqrt{x\left(x^{2}+8\right)(8-x)}}{x+1} $$ th… |
| [029](samples/029-row-16595.md) | 16595 | 16,124 | cn_contest | 8. Let the lines $l_{1} / / l_{2}$, and take 10 points $A_{1}, A_{2}, \cdots, A_{10}$ and $B_{1}, B_… |
| [030](samples/030-row-03892.md) | 3892 | 16,121 | olympiads | 3. Find $g(2021)$, if for any real $x, y$ the equality holds $$ g(x-y)=2021(g(x)+g(y))-2022 x y $$ |
| [031](samples/031-row-08214.md) | 8214 | 16,118 | cn_contest | 1. If $\frac{20122012 \cdots 201215}{n \uparrow}$ is divisible by 15, then the minimum value of $n$ … |
| [032](samples/032-row-11335.md) | 11335 | 16,118 | olympiads | G3.4 In Figure $4, A D, B C$ and $C D$ are tangents to the circle with centre at $O$ and diameter $A… |
| [033](samples/033-row-17606.md) | 17606 | 16,100 | olympiads | 3-ча 1. In a convex 13-sided polygon, all diagonals are drawn. They divide it into polygons. Let's t… |
| [034](samples/034-row-02917.md) | 2917 | 16,092 | aops_forum | For two real numbers $ a$, $ b$, with $ ab\neq 1$, define the $ \ast$ operation by \&#91; a\ast b=\frac{… |
| [035](samples/035-row-20382.md) | 20382 | 16,090 | aops_forum | In an increasing sequence of four positive integers, the first three terms form an arithmetic progre… |
| [036](samples/036-row-18834.md) | 18834 | 16,078 | aops_forum | Find the greatest possible value of $ sin(cos x) \plus{} cos(sin x)$ and determine all real numbers … |
| [037](samples/037-row-13743.md) | 13743 | 16,064 | olympiads_ref | Find all functions $f: \mathbb{Z} \rightarrow \mathbb{Z}$ that satisfy $$ f(-f(x)-f(y))=1-x-y $$ for… |
| [038](samples/038-row-14801.md) | 14801 | 16,060 | olympiads | 16-2 In the same plane, line $l_{2}$ intersects with line $l_{1}$, and line $l_{3}$ is parallel to l… |
| [039](samples/039-row-19240.md) | 19240 | 16,059 | olympiads | ## Problem Statement Find the derivative. $y=\frac{2 x-1}{4 x^{2}-4 x+3}+\frac{1}{\sqrt{2}} \cdot \o… |
| [040](samples/040-row-04929.md) | 4929 | 16,045 | olympiads | 8. Let $A=100101102103 \cdots 798799$ be a 2100-digit positive integer, formed by sequentially conne… |
| [041](samples/041-row-12743.md) | 12743 | 16,038 | olympiads | 21 Find the integer part of $$ \frac{1}{\frac{1}{2003}+\frac{1}{2004}+\frac{1}{2005}+\frac{1}{2006}+… |
| [042](samples/042-row-19560.md) | 19560 | 16,031 | olympiads | 4. A rectangle of 19 cm $\times$ 65 cm is divided by lines parallel to its sides into squares with a… |
| [043](samples/043-row-10045.md) | 10045 | 16,027 | olympiads | 9. Given a positive integer $n \geqslant 3$, the function $f:\{1,2, \cdots, n\} \rightarrow\{1,2, \c… |
| [044](samples/044-row-06156.md) | 6156 | 16,026 | aops_forum | Let $x_1,x_2,\cdots,x_n$ be postive real numbers such that $x_1x_2\cdots x_n=1$ ,$S=x^3_1+x^3_2+\cdo… |
| [045](samples/045-row-10885.md) | 10885 | 16,009 | olympiads | 5. (10 points) Among the divisors of a four-digit palindrome, exactly 3 are prime, and 39 are not pr… |
| [046](samples/046-row-19247.md) | 19247 | 16,006 | cn_contest | 3. As shown in Figure 1, the three vertices of the equilateral $\triangle D E F$ with side length $n… |
| [047](samples/047-row-11942.md) | 11942 | 16,003 | olympiads | H4. The diagram shows a right-angled triangle $A C D$ with a point $B$ on the side $A C$. The sides … |
| [048](samples/048-row-00559.md) | 559 | 15,978 | olympiads | # 8. Variant 1. 101 natural numbers are written in a circle. It is known that among any 5 consecutiv… |
| [049](samples/049-row-16403.md) | 16403 | 15,969 | olympiads | Example 21 Let $x_{1}, x_{2}, \cdots, x_{n}$ all be natural numbers, and satisfy $x_{1}+x_{2}+\cdots… |
| [050](samples/050-row-02675.md) | 2675 | 15,963 | cn_contest | One, (20 points) In the same coordinate plane, the quadratic function $y=x^{2}-x+a$ intersects with … |
| [051](samples/051-row-02542.md) | 2542 | 15,938 | aops_forum | Consider $n$ lamps clockwise numbered from $1$ to $n$ on a circle. Let $\xi$ to be a configuration w… |
| [052](samples/052-row-02648.md) | 2648 | 15,925 | olympiads | 8th Chinese 1993 Problem A2 α &gt; 0 is real and n is a positive integer. What is the maximum possible … |
| [053](samples/053-row-00586.md) | 586 | 15,923 | olympiads | (7) A three-digit number $\overline{x y z}, 1 \leqslant x \leqslant 9,0 \leqslant y, z \leqslant 9$,… |
| [054](samples/054-row-08319.md) | 8319 | 15,915 | aops_forum | Solve for $a, b, c$ given that $a \le b \le c$, and $a+b+c=-1$ $ab+bc+ac=-4$ $abc=-2$ |
| [055](samples/055-row-16339.md) | 16339 | 15,908 | aops_forum | A &#91;i&#93;quadratic&#91;/i&#93; number is a real root of the equations $ax^2 + bx + c = 0$ where $&#124;a&#124;,&#124;b&#124;,&#124;c&#124;\in\… |
| [056](samples/056-row-09289.md) | 9289 | 15,902 | olympiads | 10.9. When dividing the polynomial $x^{1951}-1$ by $x^{4}+x^{3}+2 x^{2}+$ $+x+1$, a quotient and a r… |
| [057](samples/057-row-08069.md) | 8069 | 15,895 | cn_contest | Four, for what real number $x$ does $y=x^{2}-x+1+$ $\sqrt{2(x+3)^{2}+2\left(x^{2}-5\right)^{2}}$ hav… |
| [058](samples/058-row-13884.md) | 13884 | 15,893 | aops_forum | Consider the following array: \&#91; 3, 5\\3, 8, 5\\3, 11, 13, 5\\3, 14, 24, 18, 5\\3, 17, 38, 42, 23, 5… |
| [059](samples/059-row-01463.md) | 1463 | 15,889 | olympiads | Example 4 Given ten points in space, where no four points lie on the same plane. Connect some of the… |
| [060](samples/060-row-09077.md) | 9077 | 15,889 | olympiads | 10th Australian 1989 Problem B3 Let N be the positive integers. The function f : N → N satisfies f(1… |
| [061](samples/061-row-13096.md) | 13096 | 15,885 | olympiads | ## Task Condition Find the derivative. $y=\arccos \frac{x^{2}-4}{\sqrt{x^{4}+16}}$ |
| [062](samples/062-row-06157.md) | 6157 | 15,883 | olympiads | 8. The sequence $\left\{a_{n}\right\}$ with all terms being positive integers is defined as follows:… |
| [063](samples/063-row-04671.md) | 4671 | 15,882 | cn_contest | 6. In the right triangle $\triangle A B C$, the right-angle sides $A C=3$ and $B C=4$. Line $l$ inte… |
| [064](samples/064-row-11439.md) | 11439 | 15,878 | cn_contest | 4. The condition that can determine quadrilateral $ABCD$ is a rhombus is ( ). (A) Diagonal $AC$ bise… |
| [065](samples/065-row-16925.md) | 16925 | 15,832 | olympiads | 5. In a certain social event, it was originally planned that each pair of people would shake hands e… |
| [066](samples/066-row-10771.md) | 10771 | 15,831 | olympiads | ## Task A-2.5. Let $n$ be a natural number. If a regular $n$-gon is divided into $n-2$ triangles by … |
| [067](samples/067-row-10867.md) | 10867 | 15,828 | cn_contest | 6. The lengths of the three altitudes of a triangle are $3, 4, 5$. When all three sides are the smal… |
| [068](samples/068-row-07700.md) | 7700 | 15,826 | aops_forum | Given are positive integers $ n&gt;1$ and $ a$ so that $ a&gt;n^2$, and among the integers $ a\plus{}1, a\… |
| [069](samples/069-row-09757.md) | 9757 | 15,811 | inequalities | 4. 183 Find the number of real pairs $(a, b)$ that make the system of equations $\left\{\begin{array… |
| [070](samples/070-row-13910.md) | 13910 | 15,809 | olympiads | 34 Using the digits $0,1,2,3$ and 4, find the number of 13 -digit sequences that can be written so t… |
| [071](samples/071-row-18142.md) | 18142 | 15,802 | aops_forum | Let $G$ be the centroid of triangle $ABC$ with $AB=13,BC=14,CA=15$. Calculate the sum of the distanc… |
| [072](samples/072-row-14010.md) | 14010 | 15,800 | number_theory | 28 Find the smallest positive integer $n$, such that in decimal notation $n^{3}$ ends with the digit… |
| [073](samples/073-row-02614.md) | 2614 | 15,797 | olympiads | Find all functions $f: \mathbb{N} \rightarrow \mathbb{N}$ such that for any natural number $n$, $$ f… |
| [074](samples/074-row-07053.md) | 7053 | 15,795 | olympiads | 9. (20 points) Inside an acute triangle $A B C$, a point $M$ is marked. The lines $A M, B M$, $C M$ … |
| [075](samples/075-row-02324.md) | 2324 | 15,789 | olympiads | ## Problem B1 A 98 x 98 chess board has the squares colored alternately black and white in the usual… |
| [076](samples/076-row-10592.md) | 10592 | 15,789 | olympiads | &#91; The ratio in which the bisector divides the side. &#91;The ratio of the areas of triangles with a comm… |
| [077](samples/077-row-11626.md) | 11626 | 15,785 | olympiads | 18. Let $x, y \in \mathbf{R}^{+}$, and $x+y=k$, find the range of $k$ values such that the inequalit… |
| [078](samples/078-row-15191.md) | 15191 | 15,783 | olympiads_ref | Find all pairs $(a, b)$ of positive integers such that $$ 11 a b \leq a^{3}-b^{3} \leq 12 a b $$ |
| [079](samples/079-row-13116.md) | 13116 | 15,782 | aops_forum | A wishing well is located at the point $(11,11)$ in the $xy$-plane. Rachelle randomly selects an int… |
| [080](samples/080-row-18751.md) | 18751 | 15,779 | olympiads | 3.22. The volume of the parallelepiped $A B C D A_{1} B_{1} C_{1} D_{1}$ is $V$. Find the volume of … |
| [081](samples/081-row-15370.md) | 15370 | 15,776 | olympiads | 4. &#91;4&#93; Let $A B C D$ be an isosceles trapezoid such that $A B=10, B C=15, C D=28$, and $D A=15$. The… |
| [082](samples/082-row-18488.md) | 18488 | 15,773 | cn_contest | 2. Given $$ \left(2 x+\sqrt{4 x^{2}+1}\right)\left(\sqrt{y^{2}+4}-2\right) \geqslant y&gt;0 \text {. } … |
| [083](samples/083-row-09461.md) | 9461 | 15,771 | olympiads | A convex quadrilateral's two opposite sides are each divided into 100 equal parts, and the points of… |
| [084](samples/084-row-13758.md) | 13758 | 15,727 | olympiads | XI OM - I - Task 3 Each side of a triangle with a given area $ S $ is divided into three equal parts… |
| [085](samples/085-row-03771.md) | 3771 | 15,715 | olympiads | 25) Giovanni wants to draw a square made up of nine cells (three cells per side) and write in each c… |
| [086](samples/086-row-01798.md) | 1798 | 15,712 | olympiads | 4. Determine all prime numbers $p$ and $q$ that satisfy the equation $$ (p+q)^{p}=(q-p)^{2 q-1} $$ |
| [087](samples/087-row-00919.md) | 919 | 15,710 | olympiads | A9. A unit square has an equilateral triangle drawn inside it, with a common edge. Four of these squ… |
| [088](samples/088-row-01036.md) | 1036 | 15,709 | olympiads | 20. Determine the number of ordered pairs $(x, y)$ of positive integers satisfying the following equ… |
| [089](samples/089-row-16752.md) | 16752 | 15,707 | aops_forum | Let $a_0$, $a_1$, $a_2$, $\dots$ be an infinite sequence of real numbers such that $a_0 = \frac{4}{5… |
| [090](samples/090-row-17968.md) | 17968 | 15,705 | aops_forum | Given mobile points $P(0,\ \sin \theta),\ Q(8\cos \theta,\ 0)\ \left(0\leq \theta \leq \frac{\pi}{2}… |
| [091](samples/091-row-06838.md) | 6838 | 15,698 | olympiads | $5 \cdot 23$ In a carriage, any $m(m \geqslant 3)$ passengers have a unique common friend (if A is a… |
| [092](samples/092-row-14142.md) | 14142 | 15,694 | olympiads | Problem 2. In a $3 \times 3$ table, natural numbers (not necessarily distinct) are placed such that … |
| [093](samples/093-row-04911.md) | 4911 | 15,687 | olympiads | 18. There are 2012 students standing in a row, numbered from left to right as $1, 2, \cdots \cdots 2… |
| [094](samples/094-row-17139.md) | 17139 | 15,684 | aops_forum | In the blackboard there are drawn $25$ points as shown in the figure. Gastón must choose $4$ points … |
| [095](samples/095-row-18687.md) | 18687 | 15,683 | olympiads | 4. In $\triangle ABC$, $AB=AC$, $\angle BAC=100^{\circ}$. Inside $\triangle ABC$, a point $M$ is tak… |
| [096](samples/096-row-20292.md) | 20292 | 15,675 | olympiads | 【Question 9】 As shown in the figure, four identical rectangles are arranged to form a large square w… |
| [097](samples/097-row-07764.md) | 7764 | 15,668 | olympiads | 2. The cells of a $100 \times 100$ table are painted white. In one move, it is allowed to select any… |
| [098](samples/098-row-02902.md) | 2902 | 15,661 | aops_forum | Determine all polynomials $P(x) \in \mathbb R&#91;x&#93;$ satisfying the following two conditions : (a) $P(2… |
| [099](samples/099-row-03316.md) | 3316 | 15,656 | olympiads | For any positive integer $n$, define $$ S(n)=\left&#91;\frac{n}{10^{&#91;\lg n&#93;}}\right&#93;+10\left(n-10^{&#91;\lg … |
| [100](samples/100-row-11693.md) | 11693 | 15,643 | cn_contest | 10. (14 points) As shown in Figure 1, $A$ is the right vertex of the hyperbola $\frac{x^{2}}{4}-y^{2… |
