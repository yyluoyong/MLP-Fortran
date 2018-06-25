!* 该模块定义了神经网络各层用到的数组结构，
!* 具体的参数意义参见PDF文档。
module mod_NNParameter
use mod_Precision
use mod_BaseActivationFunction
implicit none
    !---------------------------------------------------------
    ! 层与层之间的权重
    !---------------------------------------------------------
    type :: Layer_Weight
        !* 注意：W连接两层的结点数目分别为 M,N
        !*       则W为 N×M 的矩阵.
        real(kind=PRECISION), dimension(:,:), allocatable :: W    
    end type
    !=========================================================

    
    
    !---------------------------------------------------------
    ! 层的阈值
    !---------------------------------------------------------
    type :: Layer_Threshold
        !* 数组的大小是该阈值对应层的节点数目
        real(kind=PRECISION), dimension(:), allocatable :: Theta     
    end type
    !=========================================================

    
    
    !---------------------------------------------------------
    ! 层中用到的局部数组：输入、输出等
    !---------------------------------------------------------
    type :: Layer_Local_Array
        !* 数组的大小是该阈值对应层的节点数目
        !* S是输入数组，R=S-theta，Z是输出数组，Z=f(R)=f(S-Theta)
        real(kind=PRECISION), dimension(:), allocatable :: S
        real(kind=PRECISION), dimension(:), allocatable :: R
        real(kind=PRECISION), dimension(:), allocatable :: Z
        
        !* (Gamma^{k+1} W^{k+1})^T ... (Gamma^{n} W^{n}) p_zeta/p_zn
        real(kind=PRECISION), dimension(:), allocatable :: d_Matrix_part
        
        !* 以下zeta表示误差函数。
        !* zeta对W的导数
        !* 数组行、列大小分别是该权重W连接的两层的节点数目
        real(kind=PRECISION), dimension(:,:), allocatable :: dW
        
        !* 所有样本zeta对W的导数的求平均
        real(kind=PRECISION), dimension(:,:), allocatable :: avg_dW
        
        !* zeta对Theta的导数
        !* 数组的大小是该阈值对应层的节点数目
        real(kind=PRECISION), dimension(:), allocatable :: dTheta   
        
        !* 所有样本zeta对Theta的导数的求平均
        real(kind=PRECISION), dimension(:), allocatable :: avg_dTheta  
        
        !* 激活函数
        !* 注：BaseActivationFunction 是抽象类，不能使用动态数组.
        class(BaseActivationFunction), pointer :: act_fun
    end type 
    !=========================================================
    
    
    !* W、Theta 数组也可以统一放到 Layer_Local_Array 结构体中，
    !* 这里单独放置是为了应对未来代码可能的修改。
    
end module