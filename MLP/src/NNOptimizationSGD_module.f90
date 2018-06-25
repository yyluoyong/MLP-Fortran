!---------------------------------------------------------!
!* ��׼����ݶ��½���(Stochastic gradient descent)       *!
!---------------------------------------------------------!
module mod_OptimizationSGD
use mod_Precision
use mod_NNStructure
use mod_BaseGradientOptimizationMethod
use mod_NNParameter
use mod_Log
implicit none

!----------------------
! �����ࣺSGD�Ż����� |
!----------------------
type, extends(BaseGradientOptimizationMethod), public :: OptimizationSGD
    !* �̳���BaseGradientOptimizationMethod��ʵ����ӿ�
	
	!---------------------------------------------!
	!* SGD �㷨ʹ�õĲ���������                  *!
	!*��Deep Learning��, Ian Goodfellow, e.t.c.  *!
	!* һ���ϵļǺ�.                             *!
	!---------------------------------------------!
	!* ����
	real(PRECISION), private :: eps = 0.01
	!* �ȶ�ʱ�Ĳ���
	real(PRECISION), private :: eps_tau = 0.001
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
	!* �޸�SGD�㷨��Ĭ�ϲ���
	procedure, public :: set_SGD_parameter => m_set_SGD_parameter	
	
	!* batchÿ����һ����Ҫ����֮
	procedure, public :: set_iterative_step => m_set_step
	
	!* ÿ���һ��batch�ĵ�������Ҫ����֮
	!* ����������Ĳ���
    procedure, public :: update_NN => m_update_NN
	
	!* ǰ������
	procedure, public :: pre_process => m_pre_process
	
	!* ������
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
	
	!* ����������Ĳ���
	subroutine m_update_NN( this )
	implicit none
		class(OptimizationSGD), intent(inout) :: this

		integer :: layer_index, l_count 
		
		l_count = this % layers_count
        
		!* ���裺һ��batch���һ������������㣬
		!* ����õ���ƽ���ݶȣ�avg_dW��avg_dTheta
		do layer_index=1, l_count
			associate (                                                           &              
                eps        => this % eps,                                         &				
				W          => this % my_NN % pt_W(layer_index) % W,               &
                Theta      => this % my_NN % pt_Theta(layer_index) % Theta,       &                
                avg_dW     => this % my_NN % pt_Layer( layer_index ) % avg_dW,    &               
                avg_dTheta => this % my_NN % pt_Layer( layer_index ) % avg_dTheta &
            )
			
			!* �� = �� - �� * ���� 
 			W     = W     - eps * avg_dW
			Theta = Theta - eps * avg_dTheta
			
			avg_dW = 0
			avg_dTheta = 0
	
			end associate
		end do 
		
		return
	end subroutine m_update_NN
	!====
	
	!* �޸�SGD�㷨��Ĭ�ϲ���
	!* �������ú���Ĳ�����Ҫ���ؼ��ֵ���
	subroutine m_set_SGD_parameter( this, eps, eps_tau )
	implicit none
		class(OptimizationSGD), intent(inout) :: this
		real(PRECISION), optional, intent(in) :: eps, eps_tau

		if (PRESENT(eps))  this % eps = eps
		
		if (PRESENT(eps_tau))  this % eps_tau = eps_tau
		
		return
	end subroutine m_set_SGD_parameter
	!====
    
	!* ��������ṹ
	subroutine m_set_NN( this, nn_structrue )
	implicit none
		class(OptimizationSGD), intent(inout) :: this
		class(NNStructure), target, intent(in) :: nn_structrue

		this % my_NN => nn_structrue
		
		this % is_set_NN_done = .true.
		
		return
	end subroutine m_set_NN
	!====
	
	!* ���õ�����ʱ�䲽������ѧϰ��
	subroutine m_set_step( this, step )
	implicit none
		class(OptimizationSGD), intent(inout) :: this
		integer, intent(in) :: step 

		!* undo
		continue
		
		return
	end subroutine m_set_step
	!====
	
	!* ǰ������
	subroutine m_pre_process( this )
	implicit none
		class(OptimizationSGD), intent(inout) :: this

		continue
		
		return
	end subroutine m_pre_process
	!====
	
	!* ������
	subroutine m_post_process( this )
	implicit none
		class(OptimizationSGD), intent(inout) :: this

		continue
		
		return
	end subroutine m_post_process
	!====
    
    !* ���������������ڴ�ռ�
    subroutine OptimizationSGD_clean_space( this )
    implicit none
        type(OptimizationSGD), intent(inout) :: this
        
        call LogInfo("OptimizationSGD: SUBROUTINE clean_space.")
        
        return
    end subroutine OptimizationSGD_clean_space
    !====
	
	
end module