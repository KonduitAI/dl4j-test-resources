
1
in_0Const*
valueB
 *  @@*
dtype0
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
5
in_1Const*
valueB*   @*
dtype0
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
@
assert_greater/GreaterGreater	in_0/read	in_1/read*
T0
B
assert_greater/ConstConst*
valueB: *
dtype0
d
assert_greater/AllAllassert_greater/Greaterassert_greater/Const*
	keep_dims( *

Tidx0
c
(assert_greater/Assert/AssertGuard/SwitchSwitchassert_greater/Allassert_greater/All*
T0

k
*assert_greater/Assert/AssertGuard/switch_tIdentity*assert_greater/Assert/AssertGuard/Switch:1*
T0

i
*assert_greater/Assert/AssertGuard/switch_fIdentity(assert_greater/Assert/AssertGuard/Switch*
T0

R
)assert_greater/Assert/AssertGuard/pred_idIdentityassert_greater/All*
T0

[
&assert_greater/Assert/AssertGuard/NoOpNoOp+^assert_greater/Assert/AssertGuard/switch_t
�
4assert_greater/Assert/AssertGuard/control_dependencyIdentity*assert_greater/Assert/AssertGuard/switch_t'^assert_greater/Assert/AssertGuard/NoOp*
T0
*=
_class3
1/loc:@assert_greater/Assert/AssertGuard/switch_t
�
/assert_greater/Assert/AssertGuard/Assert/data_0Const+^assert_greater/Assert/AssertGuard/switch_f*;
value2B0 B*Condition x > y did not hold element-wise:*
dtype0
�
/assert_greater/Assert/AssertGuard/Assert/data_1Const+^assert_greater/Assert/AssertGuard/switch_f*#
valueB Bx (in_0/read:0) = *
dtype0
�
/assert_greater/Assert/AssertGuard/Assert/data_3Const+^assert_greater/Assert/AssertGuard/switch_f*#
valueB By (in_1/read:0) = *
dtype0
�
(assert_greater/Assert/AssertGuard/AssertAssert/assert_greater/Assert/AssertGuard/Assert/Switch/assert_greater/Assert/AssertGuard/Assert/data_0/assert_greater/Assert/AssertGuard/Assert/data_11assert_greater/Assert/AssertGuard/Assert/Switch_1/assert_greater/Assert/AssertGuard/Assert/data_31assert_greater/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
�
/assert_greater/Assert/AssertGuard/Assert/SwitchSwitchassert_greater/All)assert_greater/Assert/AssertGuard/pred_id*
T0
*%
_class
loc:@assert_greater/All
�
1assert_greater/Assert/AssertGuard/Assert/Switch_1Switch	in_0/read)assert_greater/Assert/AssertGuard/pred_id*
T0*
_class
	loc:@in_0
�
1assert_greater/Assert/AssertGuard/Assert/Switch_2Switch	in_1/read)assert_greater/Assert/AssertGuard/pred_id*
T0*
_class
	loc:@in_1
�
6assert_greater/Assert/AssertGuard/control_dependency_1Identity*assert_greater/Assert/AssertGuard/switch_f)^assert_greater/Assert/AssertGuard/Assert*
T0
*=
_class3
1/loc:@assert_greater/Assert/AssertGuard/switch_f
�
'assert_greater/Assert/AssertGuard/MergeMerge6assert_greater/Assert/AssertGuard/control_dependency_14assert_greater/Assert/AssertGuard/control_dependency*
T0
*
N
S
AddAdd	in_0/read	in_1/read(^assert_greater/Assert/AssertGuard/Merge*
T0 