
6
ConstConst*
valueB
 *
dtype0
3
imagesPackConst*
T0*

axis *
N

kernelConst*a
valueXBV"H      �?      �?      �?� 7�p��?hP<��hѿ��85��v�����?�=����>V�0�D�?*
dtype0
U
Tensordot/transpose/permConst*
dtype0*%
valueB"             
X
Tensordot/transpose	TransposeimagesTensordot/transpose/perm*
Tperm0*
T0
L
Tensordot/Reshape/shapeConst*
valueB"       *
dtype0
a
Tensordot/ReshapeReshapeTensordot/transposeTensordot/Reshape/shape*
T0*
Tshape0
O
Tensordot/transpose_1/permConst*
valueB"       *
dtype0
\
Tensordot/transpose_1	TransposekernelTensordot/transpose_1/perm*
T0*
Tperm0
N
Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0
g
Tensordot/Reshape_1ReshapeTensordot/transpose_1Tensordot/Reshape_1/shape*
T0*
Tshape0
q
Tensordot/MatMulMatMulTensordot/ReshapeTensordot/Reshape_1*
transpose_a( *
transpose_b( *
T0
L
Tensordot/shapeConst*%
valueB"             *
dtype0
N
	TensordotReshapeTensordot/MatMulTensordot/shape*
T0*
Tshape0 