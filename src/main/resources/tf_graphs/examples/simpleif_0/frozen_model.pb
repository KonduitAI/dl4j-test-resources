
8
input_0Placeholder*
dtype0*
shape
:
0
input_1Placeholder*
dtype0*
shape: 
5
VariableConst*
dtype0*
valueB
 *   @
I
Variable/readIdentityVariable*
_class
loc:@Variable*
T0
:
ConstConst*
dtype0*
valueB"       
@
SumSuminput_0Const*
T0*
	keep_dims( *

Tidx0
#
LessLessSuminput_1*
T0
'
cond/pred_idIdentityLess*
T0

X
cond/truefn/SwitchSwitchinput_0cond/pred_id*
_class
loc:@input_0*
T0
a
cond/truefn/Switch_1SwitchVariable/readcond/pred_id*
_class
loc:@Variable*
T0
I
cond/truefnAddcond/truefn/Switch:1cond/truefn/Switch_1:1*
T0
Y
cond/falsefn/SwitchSwitchinput_0cond/pred_id*
_class
loc:@input_0*
T0
b
cond/falsefn/Switch_1SwitchVariable/readcond/pred_id*
_class
loc:@Variable*
T0
H
cond/falsefnSubcond/falsefn/Switchcond/falsefn/Switch_1*
T0
@

cond/MergeMergecond/falsefncond/truefn*
T0*
N
'
outputIdentity
cond/Merge*
T0 