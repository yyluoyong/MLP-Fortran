!---------------------------------------------------------!
!* From Paper��                                          *!
!*   Author: Diederik P. Kingma, Jimmy Lei Ba.           *! 
!*   Title:  ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION. *!
!*   Year:   2015.                                       *!
!---------------------------------------------------------!
module mod_OptimizationAdam
use mod_Precision
use mod_NNStructure
use mod_BaseGradientOptimizationMethod
use mod_NNParameter
use mod_Log
implicit none

!-----------------------
! �����ࣺAdam�Ż����� |
!-----------------------
type, extends(BaseGradientOptimizationMethod), public :: OptimizationAdam
    !* �̳���BaseGradientOptimizationMethod��ʵ����ӿ�
	
	!---------------------------------------------!
	!* Adam �㷨ʹ�õĲ���������                 *!
	!*��Deep Learning��, Ian Goodfellow, e.t.c.  *!
	!* һ���ϵļǺ�.                             *!
	!---------------------------------------------!
	!* ����
	real(PRECISION), private :: eps = 0.001
	!* �ع���˥������
	real(PRECISION), private :: rho_1 = 0.9
	real(PRECISION), private :: rho_2 = 0.999
	!* ˥���ʵ��ݴ�
	real(PRECISION), private :: rho_1_t 
	real(PRECISION), private :: rho_2_t
	!* ��ֵ�ȶ�С��������ֹ����С�����ȶ�
	real(PRECISION), private :: delta = 1.E-8		
	!* Ȩ�ص�һ�׾ع���(moment estimation) s
	!*       ���׾ع���(moment estimation) r
	type (Layer_Weight), dimension(:), pointer, public :: pt_W_ME_s	
	type (Layer_Weight), dimension(:), pointer, public :: pt_W_ME_r
	
	!* ��ֵ��һ�׾ع���(moment estimation) s
	!*       ���׾ع���(moment estimation) r
	type (Layer_Threshold), dimension(:), pointer, public :: pt_Theta_ME_r
	type (Layer_Threshold), dimension(:), pointer, public :: pt_Theta_ME_s	
	!---------------------------------------------!
	
	
	
	class(NNStructure), pointer, private :: my_NN
	
	!* �Ƿ�����NN
	logical, private :: is_set_NN_done = .false.
        
	!* �Ƿ��ʼ���ڴ�ռ�
	logical, private :: is_allocate_done = .false.
	
	!* ÿ������������
	integer, private :: batch_size
	
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
	!* �޸�Adam�㷨��Ĭ�ϲ���
	procedure, public :: set_Adam_parameter => m_set_Adam_parameter	
	procedure, public :: set_batch_size => m_set_batch_size
	
	!* batchÿ����һ����Ҫ����֮
	procedure, public :: set_iterative_step => m_set_step
	
	!* ÿ���һ��batch�ĵ�������Ҫ����֮
	!* ����������Ĳ���
    procedure, public :: update_NN => m_update_NN
	!* Ȩֵ����ֵһ�ס����׾ع����� 0
	procedure, public :: set_ME_zero => m_set_ME_zero
	
	
	procedure, private :: allocate_pointer   => m_allocate_pointer
    procedure, private :: allocate_memory    => m_allocate_memory
    procedure, private :: deallocate_pointer => m_deallocate_pointer
    procedure, private :: deallocate_memory  => m_deallocate_memory
	
	final :: OptimizationAdam_clean_space
	
end type OptimizationAdam
!===================
    
    !-------------------------
    private :: m_set_NN
    private :: m_update_NN
	private :: m_set_Adam_parameter
	private :: m_set_step
	
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
		class(OptimizationAdam), intent(inout) :: this

		integer :: layer_index, l_count 
		
		l_count = this % layers_count
        
		!* ���裺һ��batch���һ������������㣬
		!* ����õ����ۻ��ݶȣ�sum_dW��sum_dTheta
		do layer_index=1, l_count
			associate (                                                           &              
                eps        => this % eps,                                         &
				rho_1      => this % rho_1,                                       &
				rho_2      => this % rho_2,                                       &
				rho_1_t    => this % rho_1_t,                                     &
				rho_2_t    => this % rho_2_t,                                     &
				delta      => this % delta,                                       &
				batch_size => this % batch_size,                                  &
				W_S        => this % pt_W_ME_s( layer_index ) % W,                &
                W_R        => this % pt_W_ME_r( layer_index ) % W,                &
                Theta_S    => this % pt_Theta_ME_s( layer_index ) % Theta,        &
                Theta_R    => this % pt_Theta_ME_r( layer_index ) % Theta,        &
				W          => this % my_NN % pt_W(layer_index) % W,               &
                Theta      => this % my_NN % pt_Theta(layer_index) % Theta,       &
                dW         => this % my_NN % pt_Layer( layer_index ) % dW,        &
                dTheta     => this % my_NN % pt_Layer( layer_index ) % dTheta,    &
                sum_dW     => this % my_NN % pt_Layer( layer_index ) % sum_dW,    &               
                sum_dTheta => this % my_NN % pt_Layer( layer_index ) % sum_dTheta &
            )
		
			!* s <-- ��_1 * s + (1 - ��_1) * g
			!* r <-- ��_2 * r + (1 - ��_2) * g �� g
			W_S = rho_1 * W_S + (1 - rho_1) * sum_dW / batch_size
			W_R = rho_2 * W_R + (1 - rho_2) * sum_dW * sum_dW / batch_size
			
			Theta_S = rho_1 * Theta_S + (1 - rho_1) * sum_dTheta / batch_size
			Theta_R = rho_2 * Theta_R + &
				(1 - rho_2) * sum_dTheta * sum_dTheta / ( batch_size * batch_size ) 
			
			!* ���� = -�� * s_hat / (��(r_hat) + ��)
			!* s_hat = s / (1 - ��^t_1), r_hat = r / (1 - ��^t_2)
			dW = -eps * (W_S / (1 - rho_1_t)) / (SQRT(W_R / (1 - rho_2_t)) + delta)
 			W = W + dW
			
			dTheta = -eps * (Theta_S / (1 - rho_1_t)) / &
				(SQRT(Theta_R / (1 - rho_2_t)) + delta)
			Theta = Theta + dTheta
			
			sum_dW = 0
			sum_dTheta = 0
	
			end associate
		end do 
		
		return
	end subroutine m_update_NN
	!====
	
	!* �޸�Adam�㷨��Ĭ�ϲ���
	!* �������ú���Ĳ�����Ҫ���ؼ��ֵ���
	subroutine m_set_Adam_parameter( this, eps, rho_1, rho_2, delta )
	implicit none
		class(OptimizationAdam), intent(inout) :: this
		real(PRECISION), optional, intent(in) :: eps, rho_1, rho_2, delta

		if (PRESENT(eps))  this % eps = eps
		
		if (PRESENT(rho_1))  this % rho_1 = rho_1

		if (PRESENT(rho_2))  this % rho_2 = rho_2
		
		if (PRESENT(delta))  this % delta = delta
		
		return
	end subroutine m_set_Adam_parameter
	!====
    
	!* ��������ṹ
	subroutine m_set_NN( this, nn_structrue )
	implicit none
		class(OptimizationAdam), intent(inout) :: this
		class(NNStructure), target, intent(in) :: nn_structrue

		this % my_NN => nn_structrue
		
		this % is_set_NN_done = .true.
		
		call this % allocate_pointer()
		call this % allocate_memory()
		
		return
	end subroutine m_set_NN
	!====

	!* 
	subroutine m_set_batch_size( this, batch_size )
	implicit none
		class(OptimizationAdam), intent(inout) :: this
		integer, intent(in) :: batch_size 

		this % batch_size = batch_size
		
		return
	end subroutine m_set_batch_size
	!====	
	
	!* ���õ�����ʱ�䲽������˥�����ݴ�
	subroutine m_set_step( this, step )
	implicit none
		class(OptimizationAdam), intent(inout) :: this
		integer, intent(in) :: step 

		this % rho_1_t = (this % rho_1)**step
		this % rho_2_t = (this % rho_2)**step
		
		return
	end subroutine m_set_step
	!====
	
	
	!* Ȩֵ����ֵһ�ס����׾ع����� 0
	subroutine m_set_ME_zero( this )
	implicit none
		class(OptimizationAdam), intent(inout) :: this

		integer :: layer_index, l_count
		
		l_count = this % layers_count
        
		do layer_index=1, l_count
			this % pt_W_ME_s( layer_index ) % W = 0
			this % pt_W_ME_r( layer_index ) % W = 0
			this % pt_Theta_ME_s( layer_index ) % Theta = 0
			this % pt_Theta_ME_r( layer_index ) % Theta = 0
		end do 
		
		return
	end subroutine m_set_ME_zero
	!====
	
	
	!* ����OptimizationAdam������ָ������ռ�
    subroutine m_allocate_pointer( this )
    implicit none
        class(OptimizationAdam), intent(inout) :: this
		
		integer :: l_count
		
		if (this % is_set_NN_done == .false.) then			
			call LogErr("mod_OptimizationAdam: SUBROUTINE m_allocate_pointer, &
				is_set_NN_done is false.")			
			stop
		end if
		
		l_count = this % my_NN % layers_count
		this % layers_count = l_count
	
		allocate( this % pt_W_ME_s(l_count) )
		allocate( this % pt_W_ME_r(l_count) )
		allocate( this % pt_Theta_ME_s(l_count) )
		allocate( this % pt_Theta_ME_r(l_count) )
		
		allocate( this % layers_node_count(0:l_count) )
		
		this % layers_node_count = this % my_NN % layers_node_count
    
        call LogDebug("OptimizationAdam: SUBROUTINE m_allocate_pointer")
        
        return
    end subroutine m_allocate_pointer
    !====
	
	!* ����ÿ��������ڴ�ռ�
    subroutine m_allocate_memory( this )
    implicit none
        class(OptimizationAdam), intent(inout) :: this
		
		integer :: M, N, layer_index, l_count
        
        l_count = this % layers_count
        
		do layer_index=1, l_count
        
			M = this % layers_node_count(layer_index - 1)
			N = this % layers_node_count(layer_index)
			          
			!* undo: Fortran2003�﷨����������
            !* ע�⣺�����СΪ N��M�������� M��N.
			allocate( this % pt_W_ME_s( layer_index ) % W(N,M) )
			allocate( this % pt_W_ME_r( layer_index ) % W(N,M) )
			allocate( this % pt_Theta_ME_s( layer_index ) % Theta(N) )
			allocate( this % pt_Theta_ME_r( layer_index ) % Theta(N) )
			
        end do
    
		this % is_allocate_done = .true.
    
        call LogDebug("OptimizationAdam: SUBROUTINE m_allocate_memory")
    
        return
    end subroutine m_allocate_memory
    !====
	
	!* ����ָ�� 
    subroutine m_deallocate_pointer( this )
    implicit none
        class(OptimizationAdam), intent(inout) :: this
		
		deallocate( this % layers_node_count )
		deallocate( this % pt_W_ME_s         )
		deallocate( this % pt_W_ME_r         )
		deallocate( this % pt_Theta_ME_s     )
		deallocate( this % pt_Theta_ME_r     )
    
        return
    end subroutine m_deallocate_pointer
	!====
    
    !* �����ڴ�ռ�
    subroutine m_deallocate_memory( this )
    implicit none
        class(OptimizationAdam), intent(inout)  :: this
		
		integer :: layer_index
        
		do layer_index=1, this % layers_count
			
			deallocate( this % pt_W_ME_s( layer_index ) % W )
			deallocate( this % pt_W_ME_r( layer_index ) % W )
			deallocate( this % pt_Theta_ME_s( layer_index ) % Theta )
			deallocate( this % pt_Theta_ME_r( layer_index ) % Theta )
			
		end do
		
		call this % deallocate_pointer()
		
		this % is_allocate_done = .false.
    
        return
    end subroutine m_deallocate_memory 
    !====
    
    !* ���������������ڴ�ռ�
    subroutine OptimizationAdam_clean_space( this )
    implicit none
        type(OptimizationAdam), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("OptimizationAdam: SUBROUTINE clean_space.")
        
        return
    end subroutine OptimizationAdam_clean_space
    !====
	
	
end module