module mod_BaseLossFunction
implicit none
    
!-------------------
! 抽象类：损失函数 |
!-------------------
type, abstract, public :: BaseLossFunction

!||||||||||||    
contains   !|
!||||||||||||

    !* 损失函数
    procedure(abs_loss), deferred, public :: loss 
 
    !* 损失函数对最后一层激活函数自变量的导数
    !* 定义见PDF文档
    procedure(abs_d_loss), deferred, public :: d_loss
    
    procedure(abs_print_msg), deferred, public :: print_msg  

end type BaseLossFunction
!===================
    

!-------------------
! 抽象类：函数接口 |
!-------------------	
abstract interface   

	!* 损失函数
    !* 该函数暂时未用到
	subroutine abs_loss( this, t, y, ans )
    use mod_Precision
    import :: BaseLossFunction
	implicit none
		class(BaseLossFunction), intent(inout) :: this
		!* t 是目标输出向量，y 是网络预测向量
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: y
        real(PRECISION), intent(inout) :: ans

	end subroutine
	!====
	
	!* 损失函数对最后一层激活函数自变量的导数
	!* 返回对网络预测向量的导数
	subroutine abs_d_loss( this, t, r, z, act_fun, dloss )
    use mod_Precision
    use mod_BaseActivationFunction
    import :: BaseLossFunction
	implicit none
		class(BaseLossFunction), intent(inout) :: this
		!* t 是目标输出向量，
        !* r 是最后一层激活函数的自变量，
        !* z 是网络预测向量
        !* act_fun 是最后一层的激活函数，
        !* dloss 是损失函数对 r 的导数
		real(PRECISION), dimension(:), intent(in) :: t
		real(PRECISION), dimension(:), intent(in) :: r
        real(PRECISION), dimension(:), intent(in) :: z
        class(BaseActivationFunction), pointer, intent(in) :: act_fun
        real(PRECISION), dimension(:), intent(inout) :: dloss

	end subroutine
	!==== 
    
    
	!* 输出信息
	subroutine abs_print_msg( this )
    import :: BaseLossFunction
	implicit none
		class(BaseLossFunction), intent(inout) :: this

	end subroutine
	!====

end interface
!===================
    
end module