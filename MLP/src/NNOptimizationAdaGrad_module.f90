!---------------------------------------------------------!
!* From Paper��                                          *!
!*   Author: John Duchi, Elad Hazan, Yoram Singer.       *! 
!*   Title:  Adaptive Subgradient Methods for Online     *!
!*			 Learning and Stochastic Optimization.       *!
!*   Year:   2011.                                       *!
!---------------------------------------------------------!
module mod_OptimizationAdaGrad
use mod_Precision
use mod_NNStructure
use mod_BaseGradientOptimizationMethod
use mod_NNParameter
use mod_Log
implicit none

!--------------------------
! �����ࣺAdaGrad�Ż����� |
!--------------------------
type, extends(BaseGradientOptimizationMethod), public :: OptimizationAdaGrad
    !* �̳���BaseGradientOptimizationMethod��ʵ����ӿ�
	
	!---------------------------------------------!
	!* AdaGrad �㷨ʹ�õĲ���������              *!
	!*��Deep Learning��, Ian Goodfellow, e.t.c.  *!
	!* һ���ϵļǺ�.                             *!
	!---------------------------------------------!
	!* ����
	real(PRECISION), private :: eps = 0.001
	!* ��ֵ�ȶ�С��������ֹ����С�����ȶ�
	real(PRECISION), private :: delta = 1.E-7
	!* Ȩֵ�ۻ�ƽ���ݶ� r
	!*       ���׾ع���(moment estimation) r
	type (Layer_Weight), dimension(:), pointer, public :: pt_W_ME_r
	
	!* ��ֵ���ۻ�ƽ���ݶ� r
	!*       ���׾ع���(moment estimation) r
	type (Layer_Threshold), dimension(:), pointer, public :: pt_Theta_ME_r
	!---------------------------------------------!
	
	
	
	class(NNStructure), pointer, private :: my_NN
	
	!* �Ƿ�����NN
	logical, private :: is_set_NN_done = .false.
        
	!* �Ƿ��ʼ���ڴ�ռ�
	logical, private :: is_allocate_done = .false.
	
	!* �����Ŀ�����������
	integer, private :: layers_count
    
	! ÿ��ڵ���Ŀ���ɵ�����: 
	!     ����Ĵ�С�����в����Ŀ��������㣩
	integer, dimension(:), allocatable, private :: layers_node_count
	
!||||||||||||    
contains   !|
!||||||||||||

	!* ��������ṹ
    procedure, public :: set_NN => m_set_NN
	
	!* ѵ��֮ǰ����
	!* �޸� AdaGrad �㷨��Ĭ�ϲ���
	procedure, public :: set_AdaGrad_parameter => m_set_AdaGrad_parameter	
	
	!* batchÿ����һ����Ҫ����֮
	procedure, public :: set_iterative_step => m_set_step
	
	!* ÿ���һ��batch�ĵ�������Ҫ����֮
	!* ����������Ĳ���
    procedure, public :: update_NN => m_update_NN
	!* Ȩֵ����ֵ���׾ع����� 0
	procedure, public :: set_ME_zero => m_set_ME_zero
	
	!* ǰ��������
	procedure, public :: pre_process => m_pre_process
	
	!* ��������
	procedure, public :: post_process => m_post_process
	
	
	procedure, private :: allocate_pointer   => m_allocate_pointer
    procedure, private :: allocate_memory    => m_allocate_memory
    procedure, private :: deallocate_pointer => m_deallocate_pointer
    procedure, private :: deallocate_memory  => m_deallocate_memory
	
	final :: OptimizationAdaGrad_clean_space
	
end type OptimizationAdaGrad
!===================
    
    !-------------------------
    private :: m_set_NN
    private :: m_update_NN
	private :: m_set_AdaGrad_parameter
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
	
	!* ����������Ĳ���
	subroutine m_update_NN( this )
	implicit none
		class(OptimizationAdaGrad), intent(inout) :: this

		integer :: layer_index, l_count 
		
		l_count = this % layers_count
        
		!* ���裺һ��batch���һ������������㣬
		!* ����õ���ƽ���ݶȣ�avg_dW��avg_dTheta
		do layer_index=1, l_count
			associate (                                                           &              
                eps        => this % eps,                                         &
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
		
			!* r <-- r + g �� g
			W_R     = W_R     + avg_dW * avg_dW 			
			Theta_R = Theta_R + avg_dTheta * avg_dTheta
			
			!* ���� = -�� / (�� + ��(r)) �� g
			dW = -eps * avg_dW / (delta + SQRT(W_R))
 			W = W + dW
			
			dTheta = -eps * avg_dTheta / (delta + SQRT(Theta_R))
			Theta = Theta + dTheta
			
			avg_dW = 0
			avg_dTheta = 0
	
			end associate
		end do 
		
		return
	end subroutine m_update_NN
	!====
	
	!* �޸� AdaGrad �㷨��Ĭ�ϲ���
	!* �������ú���Ĳ�����Ҫ���ؼ��ֵ���
	subroutine m_set_AdaGrad_parameter( this, eps, delta )
	implicit none
		class(OptimizationAdaGrad), intent(inout) :: this
		real(PRECISION), optional, intent(in) :: eps, delta

		if (PRESENT(eps))  this % eps = eps
		
		if (PRESENT(delta))  this % delta = delta
		
		return
	end subroutine m_set_AdaGrad_parameter
	!====
    
	!* ��������ṹ
	subroutine m_set_NN( this, nn_structrue )
	implicit none
		class(OptimizationAdaGrad), intent(inout) :: this
		class(NNStructure), target, intent(in) :: nn_structrue

		this % my_NN => nn_structrue
		
		this % is_set_NN_done = .true.
		
		call this % allocate_pointer()
		call this % allocate_memory()
		
		return
	end subroutine m_set_NN
	!====
	
	!* ���õ�����ʱ�䲽������˥�����ݴ�
	subroutine m_set_step( this, step )
	implicit none
		class(OptimizationAdaGrad), intent(inout) :: this
		integer, intent(in) :: step 

		continue
		
		return
	end subroutine m_set_step
	!====
	
	!* ǰ��������
	subroutine m_pre_process( this )
	implicit none
		class(OptimizationAdaGrad), intent(inout) :: this

		call this % set_ME_zero()
		
		return
	end subroutine m_pre_process
	!====
	
	!* ��������
	subroutine m_post_process( this )
	implicit none
		class(OptimizationAdaGrad), intent(inout) :: this

		continue
		
		return
	end subroutine m_post_process
	!====
	
	!* Ȩֵ����ֵһ�ס����׾ع����� 0
	subroutine m_set_ME_zero( this )
	implicit none
		class(OptimizationAdaGrad), intent(inout) :: this

		integer :: layer_index, l_count
		
		l_count = this % layers_count
        
		do layer_index=1, l_count
			this % pt_W_ME_r( layer_index ) % W = 0
			this % pt_Theta_ME_r( layer_index ) % Theta = 0
		end do 
		
		return
	end subroutine m_set_ME_zero
	!====
	
	
	!* ���� OptimizationAdaGrad ������ָ������ռ�
    subroutine m_allocate_pointer( this )
    implicit none
        class(OptimizationAdaGrad), intent(inout) :: this
		
		integer :: l_count
		
		if (this % is_set_NN_done == .false.) then			
			call LogErr("mod_OptimizationAdaGrad: SUBROUTINE m_allocate_pointer, &
				is_set_NN_done is false.")			
			stop
		end if
		
		l_count = this % my_NN % layers_count
		this % layers_count = l_count
	
		allocate( this % pt_W_ME_r(l_count) )
		allocate( this % pt_Theta_ME_r(l_count) )
		
		allocate( this % layers_node_count(0:l_count) )
		
		this % layers_node_count = this % my_NN % layers_node_count
    
        call LogDebug("OptimizationAdaGrad: SUBROUTINE m_allocate_pointer")
        
        return
    end subroutine m_allocate_pointer
    !====
	
	!* ����ÿ��������ڴ�ռ�
    subroutine m_allocate_memory( this )
    implicit none
        class(OptimizationAdaGrad), intent(inout) :: this
		
		integer :: M, N, layer_index, l_count
        
        l_count = this % layers_count
        
		do layer_index=1, l_count
        
			M = this % layers_node_count(layer_index - 1)
			N = this % layers_node_count(layer_index)
			          
			!* undo: Fortran2003�﷨����������
            !* ע�⣺�����СΪ N��M�������� M��N.
			allocate( this % pt_W_ME_r( layer_index ) % W(N,M) )
			allocate( this % pt_Theta_ME_r( layer_index ) % Theta(N) )
			
        end do
    
		this % is_allocate_done = .true.
    
        call LogDebug("OptimizationAdaGrad: SUBROUTINE m_allocate_memory")
    
        return
    end subroutine m_allocate_memory
    !====
	
	!* ����ָ�� 
    subroutine m_deallocate_pointer( this )
    implicit none
        class(OptimizationAdaGrad), intent(inout) :: this
		
		deallocate( this % layers_node_count )
		deallocate( this % pt_W_ME_r         )
		deallocate( this % pt_Theta_ME_r     )
    
        return
    end subroutine m_deallocate_pointer
	!====
    
    !* �����ڴ�ռ�
    subroutine m_deallocate_memory( this )
    implicit none
        class(OptimizationAdaGrad), intent(inout)  :: this
		
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
    
    !* ���������������ڴ�ռ�
    subroutine OptimizationAdaGrad_clean_space( this )
    implicit none
        type(OptimizationAdaGrad), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("OptimizationAdaGrad: SUBROUTINE clean_space.")
        
        return
    end subroutine OptimizationAdaGrad_clean_space
    !====
	
	
end module