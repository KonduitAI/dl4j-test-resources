
.
in_0Const*
value	B :*
dtype0
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
.
in_1Const*
value	B :*
dtype0
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
@
assert_greater/GreaterGreater	in_0/read	in_1/read*
T0
=
assert_greater/ConstConst*
valueB *
dtype0
d
assert_greater/AllAllassert_greater/Greaterassert_greater/Const*
	keep_dims( *

Tidx0
v
#assert_greater/Assert/Assert/data_0Const*;
value2B0 B*Condition x > y did not hold element-wise:*
dtype0
^
#assert_greater/Assert/Assert/data_1Const*#
valueB Bx (in_0/read:0) = *
dtype0
^
#assert_greater/Assert/Assert/data_3Const*#
valueB By (in_1/read:0) = *
dtype0
�
assert_greater/Assert/AssertAssertassert_greater/All#assert_greater/Assert/Assert/data_0#assert_greater/Assert/Assert/data_1	in_0/read#assert_greater/Assert/Assert/data_3	in_1/read*
T	
2*
	summarize
H
AddAdd	in_0/read	in_1/read^assert_greater/Assert/Assert*
T0 