module mod_BaseGradientOptimizationMethod
use mod_NNStructure
implicit none
    
!-----------------------
! 抽象类：梯度优化算法 |
!-----------------------
type, abstract, public :: BaseGradientOptimizationMethod

!||||||||||||    
contains   !|
!||||||||||||

    !* 设置网络结构
    procedure(abs_set_NN), deferred, public :: set_NN
	
	!* 设置迭代的时间步，因为学习率可能与时间相关
	procedure(abs_set_iterative_step), deferred, public :: set_iterative_step
	
	!* 更新神经网络的参数
	procedure(abs_update_NN), deferred, public :: update_NN
    

end type BaseGradientOptimizationMethod
!===================
    

!-------------------
! 抽象类：函数接口 |
!-------------------	
abstract interface   

	!* 设置网络结构
	subroutine abs_set_NN( this, nn_structrue )
    import :: BaseGradientOptimizationMethod
    import :: NNStructure
	implicit none
		class(BaseGradientOptimizationMethod), intent(inout) :: this
		class(NNStructure), target, intent(in) :: nn_structrue

	end subroutine
	!====
	
	!* 更新神经网络的参数
	subroutine abs_update_NN( this )
    import :: BaseGradientOptimizationMethod
	implicit none
		class(BaseGradientOptimizationMethod), intent(inout) :: this

	end subroutine
	!====
	
	!* 设置迭代的时间步
	subroutine abs_set_iterative_step( this, step )
    import :: BaseGradientOptimizationMethod
	implicit none
		class(BaseGradientOptimizationMethod), intent(inout) :: this
		integer, intent(in) :: step

	end subroutine
	!====

end interface
!===================
    
end module