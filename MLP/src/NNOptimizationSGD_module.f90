!---------------------------------------------------------!
!* 标准随机梯度下降法(Stochastic gradient descent)       *!
!---------------------------------------------------------!
module mod_OptimizationSGD
use mod_Precision
use mod_NNStructure
use mod_BaseGradientOptimizationMethod
use mod_NNParameter
use mod_Log
implicit none

!----------------------
! 工作类：SGD优化方法 |
!----------------------
type, extends(BaseGradientOptimizationMethod), public :: OptimizationSGD
    !* 继承自BaseGradientOptimizationMethod并实现其接口
	
	!---------------------------------------------!
	!* SGD 算法使用的参数，采用                  *!
	!*《Deep Learning》, Ian Goodfellow, e.t.c.  *!
	!* 一书上的记号.                             *!
	!---------------------------------------------!
	!* 步长
	real(PRECISION), private :: eps = 0.01
	!* 稳定时的步长
	real(PRECISION), private :: eps_tau = 0.001
	!---------------------------------------------!	
	
	class(NNStructure), pointer, private :: my_NN
	
	!* 是否设置NN
	logical, private :: is_set_NN_done = .false.
        
	!* 是否初始化内存空间
	logical, private :: is_allocate_done = .false.
	
	!* 层的数目，不含输入层
	integer, private :: layers_count
    
	! 每层节点数目构成的数组: 
	!     数组的大小是所有层的数目（含输入层）
	integer, dimension(:), allocatable, private :: layers_node_count
	
!||||||||||||    
contains   !|
!||||||||||||

	!* 设置网络结构
    procedure, public :: set_NN => m_set_NN
	
	!* 训练之前设置
	!* 修改SGD算法的默认参数
	procedure, public :: set_SGD_parameter => m_set_SGD_parameter	
	
	!* batch每迭代一次需要调用之
	procedure, public :: set_iterative_step => m_set_step
	
	!* 每完成一组batch的迭代，需要调用之
	!* 更新神经网络的参数
    procedure, public :: update_NN => m_update_NN
	
	!* 前处理工作
	procedure, public :: pre_process => m_pre_process
	
	!* 后处理工作
	procedure, public :: post_process => m_post_process
	
	final :: OptimizationSGD_clean_space
	
end type OptimizationSGD
!===================
    
    !-------------------------
    private :: m_set_NN
    private :: m_update_NN
	private :: m_set_SGD_parameter
	private :: m_set_step
	
	private :: m_pre_process
	private :: m_post_process
    !-------------------------
	
!||||||||||||    
contains   !|
!|||||||||||| 
	
	!* 更新神经网络的参数
	subroutine m_update_NN( this )
	implicit none
		class(OptimizationSGD), intent(inout) :: this

		integer :: layer_index, l_count 
		
		l_count = this % layers_count
        
		!* 假设：一个batch完成一次完整反向计算，
		!* 计算得到了平均梯度：avg_dW、avg_dTheta
		do layer_index=1, l_count
			associate (                                                           &              
                eps        => this % eps,                                         &				
				W          => this % my_NN % pt_W(layer_index) % W,               &
                Theta      => this % my_NN % pt_Theta(layer_index) % Theta,       &                
                avg_dW     => this % my_NN % pt_Layer( layer_index ) % avg_dW,    &               
                avg_dTheta => this % my_NN % pt_Layer( layer_index ) % avg_dTheta &
            )
			
			!* θ = θ - ε * △θ 
 			W     = W     - eps * avg_dW
			Theta = Theta - eps * avg_dTheta
			
			avg_dW = 0
			avg_dTheta = 0
	
			end associate
		end do 
		
		return
	end subroutine m_update_NN
	!====
	
	!* 修改SGD算法的默认参数
	!* 单独设置后面的参数需要按关键字调用
	subroutine m_set_SGD_parameter( this, eps, eps_tau )
	implicit none
		class(OptimizationSGD), intent(inout) :: this
		real(PRECISION), optional, intent(in) :: eps, eps_tau

		if (PRESENT(eps))  this % eps = eps
		
		if (PRESENT(eps_tau))  this % eps_tau = eps_tau
		
		return
	end subroutine m_set_SGD_parameter
	!====
    
	!* 设置网络结构
	subroutine m_set_NN( this, nn_structrue )
	implicit none
		class(OptimizationSGD), intent(inout) :: this
		class(NNStructure), target, intent(in) :: nn_structrue

		this % my_NN => nn_structrue
		
		this % is_set_NN_done = .true.
		
		return
	end subroutine m_set_NN
	!====
	
	!* 设置迭代的时间步，计算学习率
	subroutine m_set_step( this, step )
	implicit none
		class(OptimizationSGD), intent(inout) :: this
		integer, intent(in) :: step 

		!* undo
		continue
		
		return
	end subroutine m_set_step
	!====
	
	!* 前处理工作
	subroutine m_pre_process( this )
	implicit none
		class(OptimizationSGD), intent(inout) :: this

		continue
		
		return
	end subroutine m_pre_process
	!====
	
	!* 后处理工作
	subroutine m_post_process( this )
	implicit none
		class(OptimizationSGD), intent(inout) :: this

		continue
		
		return
	end subroutine m_post_process
	!====
    
    !* 析构函数，清理内存空间
    subroutine OptimizationSGD_clean_space( this )
    implicit none
        type(OptimizationSGD), intent(inout) :: this
        
        call LogInfo("OptimizationSGD: SUBROUTINE clean_space.")
        
        return
    end subroutine OptimizationSGD_clean_space
    !====
	
	
end module