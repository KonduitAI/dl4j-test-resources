
.
in_0Const*
value	B :*
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
G
assert_less_equal/LessEqual	LessEqual	in_0/read	in_1/read*
T0
@
assert_less_equal/ConstConst*
valueB *
dtype0
o
assert_less_equal/AllAllassert_less_equal/LessEqualassert_less_equal/Const*
	keep_dims( *

Tidx0
z
&assert_less_equal/Assert/Assert/data_0Const*<
value3B1 B+Condition x <= y did not hold element-wise:*
dtype0
a
&assert_less_equal/Assert/Assert/data_1Const*#
valueB Bx (in_0/read:0) = *
dtype0
a
&assert_less_equal/Assert/Assert/data_3Const*#
valueB By (in_1/read:0) = *
dtype0
�
assert_less_equal/Assert/AssertAssertassert_less_equal/All&assert_less_equal/Assert/Assert/data_0&assert_less_equal/Assert/Assert/data_1	in_0/read&assert_less_equal/Assert/Assert/data_3	in_1/read*
T	
2*
	summarize
K
AddAdd	in_0/read	in_1/read ^assert_less_equal/Assert/Assert*
T0 