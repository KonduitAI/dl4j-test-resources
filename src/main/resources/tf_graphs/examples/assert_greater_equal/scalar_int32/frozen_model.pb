
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
P
!assert_greater_equal/GreaterEqualGreaterEqual	in_0/read	in_1/read*
T0
C
assert_greater_equal/ConstConst*
valueB *
dtype0
{
assert_greater_equal/AllAll!assert_greater_equal/GreaterEqualassert_greater_equal/Const*
	keep_dims( *

Tidx0
}
)assert_greater_equal/Assert/Assert/data_0Const*<
value3B1 B+Condition x >= y did not hold element-wise:*
dtype0
d
)assert_greater_equal/Assert/Assert/data_1Const*#
valueB Bx (in_0/read:0) = *
dtype0
d
)assert_greater_equal/Assert/Assert/data_3Const*#
valueB By (in_1/read:0) = *
dtype0
�
"assert_greater_equal/Assert/AssertAssertassert_greater_equal/All)assert_greater_equal/Assert/Assert/data_0)assert_greater_equal/Assert/Assert/data_1	in_0/read)assert_greater_equal/Assert/Assert/data_3	in_1/read*
T	
2*
	summarize
N
AddAdd	in_0/read	in_1/read#^assert_greater_equal/Assert/Assert*
T0 