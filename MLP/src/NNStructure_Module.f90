!* ��ģ�鶨����������ṹ���Լ�������ṹ��Ӧ�����㣬
!* �磺ǰ�򴫲��������ֵ�����򴫲����������ȡ�
!* ����Ĳ�������������μ�PDF�ĵ���
module mod_NNStructure
use mod_Precision
use mod_NNParameter
use mod_Log
use mod_BaseActivationFunction
use mod_BaseLossFunction
implicit none

    !-------------------------
    ! �����ࣺ����ṹ������ |
    !-------------------------
    type, public :: NNStructure

        ! �Ƿ��ʼ����ɵı�ʶ
        logical, private :: is_init_basic = .false.
        
        ! �Ƿ��ʼ���ڴ�ռ�
        logical, private :: is_allocate_done = .false.
        
        ! �Ƿ��ʼ��Ȩֵ������ֵ
        logical, private :: is_init_weight = .false.
        logical, private :: is_init_threshold = .false.
        
        !* �Ƿ��ʼ����ʧ����
        logical, private :: is_init_loss_fun = .false.
        
        ! �����Ŀ�����������
        integer, public :: layers_count
    
        ! ÿ��ڵ���Ŀ���ɵ�����: 
		!     ����Ĵ�С�����в����Ŀ��������㣩
        integer, dimension(:), allocatable, public :: layers_node_count

        ! ָ�����в�Ȩ�ص�ָ������: 
		!     ����Ĵ�С�����в����Ŀ����������㣩  
        type (Layer_Weight), dimension(:), pointer, public :: pt_W
    
        ! ָ�����в���ֵ��ָ������: 
		!     ����Ĵ�С�����в����Ŀ����������㣩   
        type (Layer_Threshold), dimension(:), pointer, public :: pt_Theta
    
        ! ָ�����в�ֲ�����ṹ��ָ������: 
		!     ����Ĵ�С�����в����Ŀ����������㣩   
        type (Layer_Local_Array), dimension(:), pointer, public :: pt_Layer

        ! �����Ŀ������
        real(PRECISION), dimension(:), allocatable, private :: X
        ! �����Ŀ�����
        real(PRECISION), dimension(:), allocatable, private :: t
        
        !* ��ʧ����
        class(BaseLossFunction), pointer, private :: loss_function
        
    !||||||||||||    
    contains   !|
    !||||||||||||

        procedure, public :: init_basic            => m_init_basic
        procedure, public :: get_init_basic_status => c_is_init_basic
                       
        procedure, public :: forward_propagation  => m_forward_propagation
        procedure, public :: backward_propagation => m_backward_propagation
		
		procedure, public :: calc_average_gradient     => m_calc_avg_gradient
		procedure, public :: set_average_gradient_zero => m_set_average_gradient_zero
        
        procedure, public :: set_loss_function             => m_set_loss_function
		procedure, public :: set_activation_function_layer => m_set_act_fun_layer		
        
        procedure, private :: get_all_derivative_variable => m_get_all_derivative_variable                
       
        procedure, private :: set_input_layer  => m_set_input_layer
        procedure, private :: set_output_layer => m_set_output_layer
        
        procedure, private :: init_layer_weight    => m_init_layer_weight
        procedure, private :: init_layer_threshold => m_init_layer_threshold

        
        procedure, private :: get_all_layer_local_var => m_get_all_layer_local_var        
        procedure, private :: get_all_d_Matrix_part   => m_get_all_d_Matrix_part
        
        procedure, private :: get_layer_dW => m_get_layer_dW
        procedure, private :: get_all_dW   => m_get_all_dW
        
        procedure, private :: get_layer_dTheta => m_get_layer_dTheta
        procedure, private :: get_all_dTheta   => m_get_all_dTheta
		
		procedure, private :: allocate_pointer   => m_allocate_pointer
        procedure, private :: allocate_memory    => m_allocate_memory
        procedure, private :: deallocate_pointer => m_deallocate_pointer
        procedure, private :: deallocate_memory  => m_deallocate_memory

        final :: NNStructure_clean_space

    end type NNStructure
    !--------------------------------------------------------

    
    !-------------------------
    private :: m_init_basic
    private :: c_is_init_basic
    
	private :: m_allocate_pointer
	private :: m_deallocate_pointer
    private :: m_allocate_memory
    private :: m_deallocate_memory
    
    private :: m_set_act_fun_layer
    private :: m_set_input_layer
    private :: m_set_output_layer
    
    private :: m_init_layer_weight
    private :: m_init_layer_threshold
    
    private :: m_forward_propagation
    private :: m_backward_propagation
	private :: m_calc_avg_gradient
	private :: m_set_average_gradient_zero
    private :: m_get_all_layer_local_var
    
    private :: m_get_all_d_Matrix_part
    private :: m_get_layer_dW
    private :: m_get_all_dW
    private :: m_get_layer_dTheta
    private :: m_get_all_dTheta
    
    private :: m_set_loss_function
    
    private :: m_get_all_derivative_variable
    !-------------------------

!||||||||||||    
contains   !|
!||||||||||||
    
    !* ��ʼ����
    !* (1). ������������ṹ�������ڴ�ռ�;
    !* (2). �����ʼ��Ȩֵ����ֵ.
    subroutine m_init_basic( this, l_count, l_node_count )
    implicit none
        class(NNStructure), intent(inout) :: this
        integer, intent(in) :: l_count
        integer, dimension(:), intent(in) :: l_node_count

        if( .not. this % is_init_basic ) then
            
            this % layers_count = l_count
			
			call this % allocate_pointer()
			
            this % layers_node_count = l_node_count
			
			call this % allocate_memory()

            this % is_allocate_done = .true.
            
            call this % init_layer_weight()
            call this % init_layer_threshold()
            
            this % is_init_basic = .true.
            
            call LogDebug("NNStructure: SUBROUTINE m_init_basic")
        end if

        return
    end subroutine m_init_basic
    !====
    
    !* �Ƿ��ʼ��
    subroutine c_is_init_basic( this, init )
    implicit none
        class(NNStructure), intent(inout)  :: this
        logical,            intent(out)    :: init
        
        init = this % is_init_basic

        return
    end subroutine c_is_init_basic
    !====   
    
    !* ǰ����㣬��������ֵ����������������ֵ��
    !* ������Ԥ��ֵ
	!* Tips����Ҫ��ʼ���ṹ�����ø��㼤���.
    subroutine m_forward_propagation( this, X, t, y )
    implicit none
        class(NNStructure), intent(inout) :: this
        !* X ������ֵ��t ��ʵ�������y ������Ԥ�����
        real(PRECISION), dimension(:), intent(in) :: X
        real(PRECISION), dimension(:), intent(in) :: t
        real(PRECISION), dimension(:), intent(out) :: y
        
        integer :: l_count
        
        l_count = this % layers_count
        
        call this % set_input_layer( X )
        call this % set_output_layer( t )
            
        !* ǰ����㣺�������в��еľֲ����� S��R��Z
        call this % get_all_layer_local_var()
        
        y = this % pt_Layer(l_count) % Z 
        
        call LogDebug("NNStructure: SUBROUTINE m_forward_propagation")
        
        return
    end subroutine m_forward_propagation
    !====
      
    !* ������㣬�������������������ĵ���
	!* Tips����Ҫ��ʼ���ṹ�����ø��㼤�����������ʧ����.
    subroutine m_backward_propagation( this, X, t, y )
    implicit none
        class(NNStructure), intent(inout) :: this
        !* X ������ֵ��t ��ʵ�������y ������Ԥ�����
        real(PRECISION), dimension(:), intent(in) :: X
        real(PRECISION), dimension(:), intent(in) :: t
        real(PRECISION), dimension(:), intent(out) :: y
        
        integer :: M, N, layer_index, l_count
        
        l_count = this % layers_count
        
        call this % set_input_layer( X )
        call this % set_output_layer( t )
        
        !* ǰ����㣺�������в��еľֲ����� S��R��Z
        call this % get_all_layer_local_var()
        
        !* ������㣺���е�����Ϣ
        call this % get_all_derivative_variable()
        
        y = this % pt_Layer(l_count) % Z 
        
        call LogDebug("NNStructure: SUBROUTINE m_backward_propagation")
        
        return
    end subroutine m_backward_propagation
    !====

    !* ���������ƽ���ݶ�
    !* ���һ��m_backward_propagation��������
    subroutine m_calc_avg_gradient( this, batch_size )
    implicit none
        class(NNStructure), intent(inout) :: this
        integer, intent(in) :: batch_size
        
        integer :: layer_index, l_count
        
        l_count = this % layers_count
        
        !* �Ե�����Ϣ������ƽ�����ۼ�BP�㷨��Ҫ�õ�.
        do layer_index=1, l_count        
            associate (                                                   &
                dW         => this % pt_Layer( layer_index ) % dW,        &
                dTheta     => this % pt_Layer( layer_index ) % dTheta,    &
                avg_dW     => this % pt_Layer( layer_index ) % avg_dW,    &               
                avg_dTheta => this % pt_Layer( layer_index ) % avg_dTheta &
            )
        
            avg_dW = avg_dW + dW / batch_size   
            avg_dTheta = avg_dTheta + dTheta / batch_size
            
            end associate       
        end do
        
        call LogDebug("NNStructure: SUBROUTINE m_calc_avg_gradient")
        
        return
    end subroutine m_calc_avg_gradient
    !====  
	
	
	!* ��ƽ���ݶ��� 0
	subroutine m_set_average_gradient_zero( this )
    implicit none
        class(NNStructure), intent(inout) :: this
        
        integer :: layer_index, l_count
        
        l_count = this % layers_count
        
        !* �Ե�����Ϣ������ƽ�����ۼ�BP�㷨��Ҫ�õ�.
        do layer_index=1, l_count        
            associate (                                                   &
                avg_dW     => this % pt_Layer( layer_index ) % avg_dW,    &               
                avg_dTheta => this % pt_Layer( layer_index ) % avg_dTheta &
            )
        
            avg_dW = 0  
            avg_dTheta = 0
            
            end associate       
        end do
        
        call LogDebug("NNStructure: SUBROUTINE m_set_average_gradient_zero")
        
        return
    end subroutine m_set_average_gradient_zero
    !====  
    
    !* ���������󵼱�����ֵ
    subroutine m_get_all_derivative_variable( this )
    implicit none
        class(NNStructure), intent(inout) :: this
        
        !* �������е�d_Matrix_part
        call this % get_all_d_Matrix_part()
        
        !* �������е�dW
        call this % get_all_dW()
        
        !* �������е� dTheta
        call this % get_all_dTheta()
        
        call LogDebug("NNStructure: SUBROUTINE m_get_all_derivative_variable")
        
        return
    end subroutine m_get_all_derivative_variable
    !====  
    
    !* �������в��еľֲ����� S��R��Z��
    !*      �����������㡢W��Theta�������ʼ��
    subroutine m_get_all_layer_local_var( this )
    implicit none
        class(NNStructure), intent(inout) :: this
        
        integer :: layer_index, l_count
        
        l_count = this % layers_count
        
        do layer_index=1, l_count     
            associate (                                            &              
                W       => this % pt_W(layer_index) % W,           &
                Theta   => this % pt_Theta(layer_index) % Theta,   &
                S       => this % pt_Layer(layer_index) % S,       &
                R       => this % pt_Layer(layer_index) % R,       &
                Z       => this % pt_Layer(layer_index) % Z,       &
				act_fun => this % pt_Layer(layer_index) % act_fun, &
                Z_pre   => this % pt_Layer(layer_index-1) % Z      &
            )
        
            !* S^{k} = W^{k}*Z^{k-1}
            S = MATMUL(W, Z_pre) 
            
            !* R^{k} = S^{k} - Theta^{k}
            R = S - Theta
                
            !* Z^{k} = f(R^{k})
            call act_fun % f_vect( R, Z )      
            
            end associate                                 
        end do
        
        call LogDebug("NNStructure: SUBROUTINE m_get_all_layer_local_var")
        
        return
    end subroutine m_get_all_layer_local_var
    !====
   
    !* �������е�d_Matrix_part:
    !*      Ŀ��������ѳ�ʼ��.    
    subroutine m_get_all_d_Matrix_part( this )
    implicit none
        class(NNStructure), intent(inout) :: this
        
        integer :: layer_index, l_count
        integer :: i, j
        real(PRECISION) :: df_to_dr
        !* (��^{k} W^{k})^T
        real(PRECISION), dimension(:,:), allocatable :: WT_GammaT
        
        l_count = this % layers_count
                      
        !---------------------------------
        !* (1): ������һ��
        !* zeta��rn�ĵ���
        associate (                                                  &              
            matrix_part => this % pt_Layer(l_count) % d_Matrix_part, &
            R           => this % pt_Layer(l_count) % R,             &
            Z           => this % pt_Layer(l_count) % Z,             &
            act_fun     => this % pt_Layer(l_count) % act_fun,       &
            t           => this % t                                  &
        )
        
        !* ����ϵ� d_Matrix_part Ϊ zeta��rn�ĵ���
        call this % loss_function % d_loss(t, R, Z, act_fun, matrix_part)
            
        end associate
        !---------------------------------
        
        
        !---------------------------------
        !* (2): �����ڶ���
        if (l_count > 1) then
            associate (                                                        &              
                W                => this % pt_W(l_count) % W,                  &
                matrix_part_next => this % pt_Layer(l_count) % d_Matrix_part,  &
                matrix_part      => this % pt_Layer(l_count-1) % d_Matrix_part &
            )
                
            matrix_part = MATMUL(TRANSPOSE(W), matrix_part_next)   
            
            end associate
        end if
        !---------------------------------
        
        
        !---------------------------------
        !* (3): �����
        do layer_index=l_count-2, 1, -1
            associate (                                                             &                              
                W                => this % pt_W(layer_index + 1) % W,               &
                M                => this % layers_node_count(layer_index),          &
                N                => this % layers_node_count(layer_index+1),        &
                R                => this % pt_Layer(layer_index+1) % R,             &
                matrix_part_next => this % pt_Layer(layer_index+1) % d_Matrix_part, &
                matrix_part      => this % pt_Layer(layer_index) % d_Matrix_part,   &
                act_fun          => this % pt_Layer(layer_index+1) % act_fun        &
            )
            
            allocate( WT_GammaT(M, N) )
            
			!* ��ǰ��Ϊk������(��^{k+1} W^{k+1})^T			
            do j=1, N
                call act_fun % df( j, R, df_to_dr ) 
                WT_GammaT(:, j) = W(j, :) * df_to_dr
            end do
            
            matrix_part = MATMUL(WT_GammaT, matrix_part_next)
            
            deallocate( WT_GammaT )
            
            end associate    
        end do
        !---------------------------------
        
        call LogDebug("NNStructure: SUBROUTINE m_get_all_d_Matrix_part")
            
        return
    end subroutine m_get_all_d_Matrix_part
    !====
    
    !* �������е�dW
    subroutine m_get_all_dW( this )
    implicit none
        class(NNStructure), intent(inout) :: this
        
        integer :: layer_index, l_count
        
        l_count = this % layers_count

        do layer_index=1, l_count
            call this % get_layer_dW(layer_index)
        end do
        
        call LogDebug("NNStructure: SUBROUTINE m_get_all_dW")
        
        return
    end subroutine m_get_all_dW    
    !====    
    
    !* �������е� dTheta
    subroutine m_get_all_dTheta( this )
    implicit none
        class(NNStructure), intent(inout) :: this
        
        integer :: layer_index, l_count
        
        l_count = this % layers_count

        do layer_index=1, l_count
            call this % get_layer_dTheta(layer_index)
        end do
        
        call LogDebug("NNStructure: SUBROUTINE m_get_all_dTheta")
        
        return
    end subroutine m_get_all_dTheta 
    !====       
    
    !* ����Ŀ����dW
    subroutine m_get_layer_dW( this, layer_index )
    implicit none
        class(NNStructure), intent(inout) :: this
        integer, intent(in) :: layer_index
        
        integer :: M, N
        integer :: i, j
        real(PRECISION) :: r, df_to_dr        			
        
        !* Ŀ������������С
        M = this % layers_node_count(layer_index - 1)
        !* Ŀ�����������С 
        N = this % layers_node_count(layer_index)

		!---------------------------------
		!* (1)�����һ��
		!* ���һ�㹫ʽ�������������PDF�ĵ�.
		if (layer_index == this % layers_count) then
			
			associate (                                                      &   
                W           => this % pt_W(layer_index) % W,                 &
                dW          => this % pt_Layer(layer_index) % dW,            &
                matrix_part => this % pt_Layer(layer_index) % d_Matrix_part, &
                Z_pre       => this % pt_Layer(layer_index-1) % Z            &          
			)			
			
			!* dW^{k}_{ij} = z^{k-1}_j * E_i * d_Matrix_part
			do i=1, N 
				do j=1, M
					dW(i, j) = Z_pre(j) * matrix_part(i) + 1.E-4 * W(i,j)
				end do
            end do

            !dW = dW + 1.E-4 * W
            
			end associate
			
			return
		end if	
		!---------------------------------
		
		!---------------------------------
		!* (2)�������
        associate (                                                      &  
            W           => this % pt_W(layer_index) % W,                 &
            dW          => this % pt_Layer(layer_index) % dW,            &
            R           => this % pt_Layer(layer_index) % R,             &
            matrix_part => this % pt_Layer(layer_index) % d_Matrix_part, &
			act_fun     => this % pt_Layer(layer_index) % act_fun,       &
            Z_pre       => this % pt_Layer(layer_index-1) % Z            &          
        )
        
        do i=1, N     
            call act_fun % df( i, R, df_to_dr )
            
			!* ��ǰ��Ϊk��
            !* dW^{k}_{ij} = f'(r^{k}_i) * z^{k-1}_j * E_i * d_Matrix_part
            do j=1, M
                dW(i, j) = df_to_dr * Z_pre(j) * matrix_part(i) + 1.E-4 * W(i,j)
            end do
            
            !dW = dW + 1.E-4 * W
            
        end do
        
        end associate
		!---------------------------------
        
        call LogDebug("NNStructure: SUBROUTINE m_get_layer_dW")
        
        return
    end subroutine m_get_layer_dW    
    !====
    
    !* ����Ŀ���� dTheta
    subroutine m_get_layer_dTheta( this, layer_index )
    implicit none
        class(NNStructure), intent(inout) :: this
        integer, intent(in) :: layer_index
        
        integer :: N
        integer :: i
        real(PRECISION) :: r, df_to_dr
        
        N = this % layers_node_count(layer_index)

		!---------------------------------
		!* (1)�����һ��
		!* ���һ�㹫ʽ�������������PDF�ĵ�.
		if (layer_index == this % layers_count) then
			associate (                                                     &      
                dTheta      => this % pt_Layer(layer_index) % dTheta,       &
                matrix_part => this % pt_Layer(layer_index) % d_Matrix_part &
            )
		
			!* dTheta^{k}_{i} = -E_i * d_Matrix_part
			do i=1, N   
				dTheta(i) = -matrix_part(i)
			end do
		
			end associate
			
			return
		end if
		!---------------------------------
		
		!---------------------------------
		!* (2)�������
        associate (                                                      &      
            dTheta      => this % pt_Layer(layer_index) % dTheta,        &
            R           => this % pt_Layer(layer_index) % R,             &
            matrix_part => this % pt_Layer(layer_index) % d_Matrix_part, &
			act_fun     => this % pt_Layer(layer_index) % act_fun        &
        )
        
        do i=1, N          
            call act_fun % df( i, R, df_to_dr )
        
			!* ��ǰ��Ϊk��
            !* dTheta^{k}_{i} = -f'(r^{k}_i) * E_i * d_Matrix_part
            dTheta(i) = -df_to_dr * matrix_part(i)
        end do
        
        end associate
		!---------------------------------
        
        call LogDebug("NNStructure: SUBROUTINE m_get_layer_dTheta")
        
        return
    end subroutine m_get_layer_dTheta    
    !====
 
	!* ���������
    subroutine m_set_input_layer( this, X_in )
    implicit none
        class(NNStructure), intent(inout) :: this
        real(PRECISION), dimension(:), intent(in) :: X_in 
    
        this % X = X_in
        this % pt_Layer( 0 ) % Z = X_in
        
        call LogDebug("NNStructure: SUBROUTINE m_set_input_layer")
        
        return
    end subroutine m_set_input_layer
    !====
    
    !* ���������
    subroutine m_set_output_layer( this, t )
    implicit none
        class(NNStructure), intent(inout) :: this
        real(PRECISION), dimension(:), intent(in) :: t 
        
        this % t = t
        
        call LogDebug("NNStructure: SUBROUTINE m_set_output_layer")
        
        return
    end subroutine m_set_output_layer
    !====

	!* ����ָ����ļ����
    subroutine m_set_act_fun_layer( this, layer_index, act_fun )
    implicit none
        class(NNStructure), intent(inout) :: this
        integer, intent(in) :: layer_index
        class(BaseActivationFunction), target, intent(in) :: act_fun
    
        this % pt_Layer(layer_index) % act_fun => act_fun
        
        call LogDebug("NNStructure: SUBROUTINE m_set_act_fun_layer")
        
        return
    end subroutine m_set_act_fun_layer
    !====
	
	!* ������ʧ����
    subroutine m_set_loss_function( this, loss_fun )
    use mod_CrossEntropy
    implicit none
        class(NNStructure), intent(inout) :: this
        class(BaseLossFunction), target, intent(in) :: loss_fun
        
        this % loss_function => loss_fun        
        
        this % is_init_loss_fun = .true.
        
        call LogDebug("NNStructure: SUBROUTINE m_set_loss_function")
        
        return
    end subroutine m_set_loss_function
    !====

	!* �����ʼ����ֵ��Ĭ�ϳ�ʼ����(-1,1)
    !* ��Train�����У������������ó�ʼ��.
    subroutine m_init_layer_weight( this )
    implicit none
        class(NNStructure), intent(inout) :: this
     
        integer :: layer_index, l_count
        
        l_count = this % layers_count
        
        call RANDOM_SEED()
        
        do layer_index=1, l_count
            associate (                            &              
                W => this % pt_W(layer_index) % W  &
            )
                
            call RANDOM_NUMBER(W)
            W = 2.0 * W - 1.0
            
            end associate
        end do
        
        this % is_init_weight = .true.
        
        call LogDebug("NNStructure: SUBROUTINE m_init_layer_weight.")
        
        return
    end subroutine m_init_layer_weight
    !====
    
    !* �����ʼ����ֵ��Ĭ�ϳ�ʼ����(-1,1)
    !* ��Train�����У������������ó�ʼ��.
    subroutine m_init_layer_threshold( this )
    implicit none
        class(NNStructure), intent(inout) :: this
     
        integer :: layer_index, l_count
        
        l_count = this % layers_count
        
        call RANDOM_SEED()       
        
        do layer_index=1, l_count
            associate (                                        &              
                Theta => this % pt_Theta(layer_index) % Theta  &
            )
                
            call RANDOM_NUMBER(Theta) 
            Theta = 2.0 * Theta - 1.0
            
            end associate
        end do
  
        this % is_init_threshold = .true.
        
        call LogDebug("NNStructure: SUBROUTINE m_init_layer_threshold.")
        
        return
    end subroutine m_init_layer_threshold
    !====
	
    !* ����NNStructure������ָ������ռ�
    subroutine m_allocate_pointer( this )
    implicit none
        class(NNStructure), intent(inout) :: this
		
		integer :: l_count
		
		l_count = this % layers_count
		
		!* ע���±��0��ʼ����Ϊ��һ�������
		allocate( this % layers_node_count(0:l_count) )
		
		!* ��������㲻������Щ�ṹ
		allocate( this % pt_W(l_count) )
		allocate( this % pt_Theta(l_count) )
        
        !* �����ֻ����Z^0��Ϊ�˵�������
		allocate( this % pt_Layer(0:l_count) )
    
        call LogDebug("NNStructure: SUBROUTINE m_allocate_pointer")
        
        return
    end subroutine m_allocate_pointer
    !====
    
    !* ����ÿ��������ڴ�ռ�
    subroutine m_allocate_memory( this )
    implicit none
        class(NNStructure), intent(inout) :: this
		
		integer :: M, N, layer_index, l_count
        
        l_count = this % layers_count
		
        !* �����
        M = this % layers_node_count(0)
        allocate( this % X(M) )    
        allocate( this % pt_Layer( 0 ) % Z(M) )
        
        !* �����
        M = this % layers_node_count(l_count)
        allocate( this % t(M) )
        
		do layer_index=1, l_count
        
			M = this % layers_node_count(layer_index - 1)
			N = this % layers_node_count(layer_index)
			          
			!* undo: Fortran2003�﷨����������
            !* ע�⣺�����СΪ N��M�������� M��N.
			allocate( this % pt_W( layer_index ) % W(N,M) )
			allocate( this % pt_Theta( layer_index ) % Theta(N) )
			allocate( this % pt_Layer( layer_index ) % S(N) )
			allocate( this % pt_Layer( layer_index ) % R(N) )
			allocate( this % pt_Layer( layer_index ) % Z(N) )
            allocate( this % pt_Layer( layer_index ) % d_Matrix_part(N) )
            
            !* ע�⣺�����СΪ N��M�������� M��N.
            allocate( this % pt_Layer( layer_index ) % dW(N,M) )
            allocate( this % pt_Layer( layer_index ) % avg_dW(N,M) )
            
            allocate( this % pt_Layer( layer_index ) % dTheta(N) )
            allocate( this % pt_Layer( layer_index ) % avg_dTheta(N) )
        end do
		
		call this % set_average_gradient_zero()
        
        call LogDebug("NNStructure: SUBROUTINE m_allocate_memory")
    
        return
    end subroutine m_allocate_memory
    !====
	
	!* ����ָ�� 
    subroutine m_deallocate_pointer( this )
    implicit none
        class(NNStructure), intent(inout) :: this
		
		deallocate( this % layers_node_count )
		deallocate( this % pt_W )
		deallocate( this % pt_Theta )
		deallocate( this % pt_Layer )
    
        return
    end subroutine m_deallocate_pointer
	!====
    
    !* �����ڴ�ռ�
    subroutine m_deallocate_memory( this )
    implicit none
        class(NNStructure), intent(inout)  :: this
		
		integer :: layer_index
		
        !* ����㡢�����
        deallocate( this % X )
        deallocate( this % t )
		deallocate( this % pt_Layer( 0 ) % Z )
        
		do layer_index=1, this%layers_count
			
			deallocate( this % pt_W( layer_index ) % W )
			deallocate( this % pt_Theta( layer_index ) % Theta )
			deallocate( this % pt_Layer( layer_index ) % S )
			deallocate( this % pt_Layer( layer_index ) % R )
			deallocate( this % pt_Layer( layer_index ) % Z )
            deallocate( this % pt_Layer( layer_index ) % dW )
			deallocate( this % pt_Layer( layer_index ) % dTheta )
            deallocate( this % pt_Layer( layer_index ) % avg_dW )
			deallocate( this % pt_Layer( layer_index ) % avg_dTheta )
			deallocate( this % pt_Layer( layer_index ) % d_Matrix_part )
			
		end do
		
		call this % deallocate_pointer()
    
        return
    end subroutine m_deallocate_memory 
    !====
    
    !* ���������������ڴ�ռ�
    subroutine NNStructure_clean_space( this )
    implicit none
        type(NNStructure), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("NNStructure: SUBROUTINE clean_space.")
        
        return
    end subroutine NNStructure_clean_space
    !====
    
    
end module