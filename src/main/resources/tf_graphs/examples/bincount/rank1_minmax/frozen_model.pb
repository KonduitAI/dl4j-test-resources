
Y
in_0Const*=
value4B2
"(      	                        *
dtype0
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
<
bincount/ShapeConst*
valueB:
*
dtype0
<
bincount/ConstConst*
dtype0*
valueB: 
[
bincount/ProdProdbincount/Shapebincount/Const*
	keep_dims( *

Tidx0*
T0
<
bincount/Greater/yConst*
value	B : *
dtype0
G
bincount/GreaterGreaterbincount/Prodbincount/Greater/y*
T0
O
bincount/CastCastbincount/Greater*

SrcT0
*
Truncate( *

DstT0
>
bincount/Const_1Const*
valueB: *
dtype0
V
bincount/MaxMax	in_0/readbincount/Const_1*
T0*
	keep_dims( *

Tidx0
8
bincount/add/yConst*
dtype0*
value	B :
<
bincount/addAddV2bincount/Maxbincount/add/y*
T0
9
bincount/mulMulbincount/Castbincount/add*
T0
<
bincount/minlengthConst*
value	B :*
dtype0
F
bincount/MaximumMaximumbincount/minlengthbincount/mul*
T0
<
bincount/maxlengthConst*
value	B :*
dtype0
J
bincount/MinimumMinimumbincount/maxlengthbincount/Maximum*
T0
9
bincount/Const_2Const*
valueB *
dtype0
U
bincount/BincountBincount	in_0/readbincount/Minimumbincount/Const_2*
T0 