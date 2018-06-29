!---------------------------------------------------------!
!* From Paper：                                          *!
!*   Author: Geoffrey Hinton.                            *! 
!*   Title:  Tutorial on deep learning.                  *!
!*   Year:   2012.                                       *!
!*   Other:  IPAM Graduate Summer School: Deep Learning, *! 
!*			 Feature Learning. 261                       *! 
!---------------------------------------------------------!
module mod_OptimizationRMSProp
use mod_Precision
use mod_NNStructure
use mod_BaseGradientOptimizationMethod
use mod_NNParameter
use mod_Log
implicit none

!--------------------------
! 工作类：RMSProp优化方法 |
!--------------------------
type, extends(BaseGradientOptimizationMethod), public :: OptimizationRMSProp
    !* 继承自BaseGradientOptimizationMethod并实现其接口
	
	!---------------------------------------------!
	!* RMSProp 算法使用的参数，采用              *!
	!*《Deep Learning》, Ian Goodfellow, e.t.c.  *!
	!* 一书上的记号.                             *!
	!---------------------------------------------!
	!* 步长
	real(PRECISION), private :: eps = 0.001
	!* 矩估计衰减速率
	real(PRECISION), private :: rho = 0.99
	!* 数值稳定小参数，防止除以小数不稳定
	real(PRECISION), private :: delta = 1.E-6
	!* 权值累积平方梯度 r
	!*       二阶矩估计(moment estimation) r
	type (Layer_Weight), dimension(:), pointer, public :: pt_W_ME_r
	
	!* 阈值的累积平方梯度 r
	!*       二阶矩估计(moment estimation) r
	type (Layer_Threshold), dimension(:), pointer, public :: pt_Theta_ME_r
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
	!* 修改 RMSProp 算法的默认参数
	procedure, public :: set_RMSProp_parameter => m_set_RMSProp_parameter	
	
	!* batch每迭代一次需要调用之
	procedure, public :: set_iterative_step => m_set_step
	
	!* 每完成一组batch的迭代，需要调用之
	!* 更新神经网络的参数
    procedure, public :: update_NN => m_update_NN
	!* 权值、阈值二阶矩估计置 0
	procedure, public :: set_ME_zero => m_set_ME_zero
	
	!* 前处理工作
	procedure, public :: pre_process => m_pre_process
	
	!* 后处理工作
	procedure, public :: post_process => m_post_process
	
	
	procedure, private :: allocate_pointer   => m_allocate_pointer
    procedure, private :: allocate_memory    => m_allocate_memory
    procedure, private :: deallocate_pointer => m_deallocate_pointer
    procedure, private :: deallocate_memory  => m_deallocate_memory
	
	final :: OptimizationRMSProp_clean_space
	
end type OptimizationRMSProp
!===================
    
    !-------------------------
    private :: m_set_NN
    private :: m_update_NN
	private :: m_set_RMSProp_parameter
	private :: m_set_step
	
	private :: m_set_ME_zero
	
	private :: m_pre_process
	private :: m_post_process
	
	private :: m_allocate_pointer
	private :: m_allocate_memory
	private :: m_deallocate_pointer
	private :: m_deallocate_memory
    !-------------------------
	
!||||||||||||    
contains   !|
!|||||||||||| 
	
	!* 更新神经网络的参数
	subroutine m_update_NN( this, bp_algorithm )
	implicit none
		class(OptimizationRMSProp), intent(inout) :: this
		character(len=*), optional, intent(in) :: bp_algorithm

		integer :: layer_index, l_count 
		
		l_count = this % layers_count
        
		!* 假设：一个batch完成一次完整反向计算，
		!* 计算得到了平均梯度：avg_dW、avg_dTheta
		do layer_index=1, l_count
			associate (                                                           &              
                eps        => this % eps,                                         &
				rho        => this % rho,                                         &
				delta      => this % delta,                                       &
                W_R        => this % pt_W_ME_r( layer_index ) % W,                &
                Theta_R    => this % pt_Theta_ME_r( layer_index ) % Theta,        &
				W          => this % my_NN % pt_W(layer_index) % W,               &
                Theta      => this % my_NN % pt_Theta(layer_index) % Theta,       &
                dW         => this % my_NN % pt_Layer( layer_index ) % dW,        &
                dTheta     => this % my_NN % pt_Layer( layer_index ) % dTheta,    &
                avg_dW     => this % my_NN % pt_Layer( layer_index ) % avg_dW,    &               
                avg_dTheta => this % my_NN % pt_Layer( layer_index ) % avg_dTheta &
            )
		
			!* r <-- ρ * r + (1 - ρ) * g ⊙ g
			W_R     = rho * W_R     + (1 - rho) * avg_dW * avg_dW 			
			Theta_R = rho * Theta_R + (1 - rho) * avg_dTheta * avg_dTheta
			
			!* △θ = -ε / (√(δ + r) ⊙ g
			dW = -eps * avg_dW / SQRT(delta + W_R)
 			W = W + dW
			
			dTheta = -eps * avg_dTheta / SQRT(delta + Theta_R)
			Theta = Theta + dTheta
			
			avg_dW = 0
			avg_dTheta = 0
	
			end associate
		end do 
		
		return
	end subroutine m_update_NN
	!====
	
	!* 修改 RMSProp 算法的默认参数
	!* 单独设置后面的参数需要按关键字调用
	subroutine m_set_RMSProp_parameter( this, eps, rho, delta )
	implicit none
		class(OptimizationRMSProp), intent(inout) :: this
		real(PRECISION), optional, intent(in) :: eps, rho, delta

		if (PRESENT(eps))  this % eps = eps
		
		if (PRESENT(rho))  this % rho = rho
		
		if (PRESENT(delta))  this % delta = delta
		
		return
	end subroutine m_set_RMSProp_parameter
	!====
    
	!* 设置网络结构
	subroutine m_set_NN( this, nn_structrue )
	implicit none
		class(OptimizationRMSProp), intent(inout) :: this
		class(NNStructure), target, intent(in) :: nn_structrue

		this % my_NN => nn_structrue
		
		this % is_set_NN_done = .true.
		
		call this % allocate_pointer()
		call this % allocate_memory()
		
		return
	end subroutine m_set_NN
	!====
	
	!* 设置迭代的时间步，计算衰减率幂次
	subroutine m_set_step( this, step )
	implicit none
		class(OptimizationRMSProp), intent(inout) :: this
		integer, intent(in) :: step 

		continue
		
		return
	end subroutine m_set_step
	!====
	
	!* 前处理工作
	subroutine m_pre_process( this )
	implicit none
		class(OptimizationRMSProp), intent(inout) :: this

		call this % set_ME_zero()
		
		return
	end subroutine m_pre_process
	!====
	
	!* 后处理工作
	subroutine m_post_process( this )
	implicit none
		class(OptimizationRMSProp), intent(inout) :: this

		continue
		
		return
	end subroutine m_post_process
	!====
	
	!* 权值、阈值一阶、二阶矩估计置 0
	subroutine m_set_ME_zero( this )
	implicit none
		class(OptimizationRMSProp), intent(inout) :: this

		integer :: layer_index, l_count
		
		l_count = this % layers_count
        
		do layer_index=1, l_count
			this % pt_W_ME_r( layer_index ) % W = 0
			this % pt_Theta_ME_r( layer_index ) % Theta = 0
		end do 
		
		return
	end subroutine m_set_ME_zero
	!====
	
	
	!* 申请OptimizationRMSProp包含的指针所需空间
    subroutine m_allocate_pointer( this )
    implicit none
        class(OptimizationRMSProp), intent(inout) :: this
		
		integer :: l_count
		
		if (this % is_set_NN_done == .false.) then			
			call LogErr("mod_OptimizationRMSProp: SUBROUTINE m_allocate_pointer, &
				is_set_NN_done is false.")			
			stop
		end if
		
		l_count = this % my_NN % layers_count
		this % layers_count = l_count
	
		allocate( this % pt_W_ME_r(l_count) )
		allocate( this % pt_Theta_ME_r(l_count) )
		
		allocate( this % layers_node_count(0:l_count) )
		
		this % layers_node_count = this % my_NN % layers_node_count
    
        call LogDebug("OptimizationRMSProp: SUBROUTINE m_allocate_pointer")
        
        return
    end subroutine m_allocate_pointer
    !====
	
	!* 申请每层所需的内存空间
    subroutine m_allocate_memory( this )
    implicit none
        class(OptimizationRMSProp), intent(inout) :: this
		
		integer :: M, N, layer_index, l_count
        
        l_count = this % layers_count
        
		do layer_index=1, l_count
        
			M = this % layers_node_count(layer_index - 1)
			N = this % layers_node_count(layer_index)
			          
			!* undo: Fortran2003语法检测申请错误
            !* 注意：矩阵大小为 N×M，而不是 M×N.
			allocate( this % pt_W_ME_r( layer_index ) % W(N,M) )
			allocate( this % pt_Theta_ME_r( layer_index ) % Theta(N) )
			
        end do
    
		this % is_allocate_done = .true.
    
        call LogDebug("OptimizationRMSProp: SUBROUTINE m_allocate_memory")
    
        return
    end subroutine m_allocate_memory
    !====
	
	!* 销毁指针 
    subroutine m_deallocate_pointer( this )
    implicit none
        class(OptimizationRMSProp), intent(inout) :: this
		
		deallocate( this % layers_node_count )
		deallocate( this % pt_W_ME_r         )
		deallocate( this % pt_Theta_ME_r     )
    
        return
    end subroutine m_deallocate_pointer
	!====
    
    !* 销毁内存空间
    subroutine m_deallocate_memory( this )
    implicit none
        class(OptimizationRMSProp), intent(inout)  :: this
		
		integer :: layer_index
        
		do layer_index=1, this % layers_count
			
			deallocate( this % pt_W_ME_r( layer_index ) % W )
			deallocate( this % pt_Theta_ME_r( layer_index ) % Theta )
			
		end do
		
		call this % deallocate_pointer()
		
		this % is_allocate_done = .false.
    
        return
    end subroutine m_deallocate_memory 
    !====
    
    !* 析构函数，清理内存空间
    subroutine OptimizationRMSProp_clean_space( this )
    implicit none
        type(OptimizationRMSProp), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("OptimizationRMSProp: SUBROUTINE clean_space.")
        
        return
    end subroutine OptimizationRMSProp_clean_space
    !====
	
	
end module