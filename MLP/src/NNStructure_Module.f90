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
        logical, private :: is_init = .false.
        
        ! �Ƿ��ʼ���ڴ�ռ�
        logical, private :: is_allocate_done = .false.
    
        ! �Ƿ��ʼ�������
        logical, private :: is_init_act_fun = .false.
        
        ! �Ƿ��ʼ������㡢�����
        logical, private :: is_init_input_layer = .false.
        logical, private :: is_init_output_layer = .false.
        
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
        class(BaseLossFunction), pointer :: loss_function
        
    !||||||||||||    
    contains   !|
    !||||||||||||

        procedure, public :: init_basic => m_init
        procedure, public :: get_init_status => c_isInit
                       
        procedure, public :: forward_propagation  => m_forward_propagation
        procedure, public :: backward_propagation => m_backward_propagation
        
        procedure, public :: set_loss_function => m_set_loss_function
        
        procedure, private :: get_all_derivative_variable => m_get_all_derivative_variable
                
        procedure, private :: allocate_pointer   => m_allocate_pointer
        procedure, private :: allocate_memory    => m_allocate_memory
        procedure, private :: deallocate_pointer => m_deallocate_pointer
        procedure, private :: deallocate_memory  => m_deallocate_memory
        
        procedure, public :: init_activation_function => m_init_activation_function
        
        procedure, private :: init_input_layer  => m_init_input_layer
        procedure, private :: init_output_layer => m_init_output_layer
        
        procedure, private :: init_layer_weight    => m_init_layer_weight
        procedure, private :: init_layer_threshold => m_init_layer_threshold

        
        procedure, private :: get_all_layer_local_var => m_get_all_layer_local_var        
        procedure, private :: get_all_d_Matrix_part   => m_get_all_d_Matrix_part
        
        procedure, private :: get_layer_dW => m_get_layer_dW
        procedure, private :: get_all_dW   => m_get_all_dW
        
        procedure, private :: get_layer_dTheta => m_get_layer_dTheta
        procedure, private :: get_all_dTheta   => m_get_all_dTheta

        final :: NNStructure_clean_space

    end type NNStructure
    !--------------------------------------------------------

    
    !-------------------------
    private :: m_init
    private :: c_isInit
    
	private :: m_allocate_pointer
	private :: m_deallocate_pointer
    private :: m_allocate_memory
    private :: m_deallocate_memory
    
    private :: m_init_activation_function
    private :: m_init_input_layer
    private :: m_init_output_layer
    
    private :: m_init_layer_weight
    private :: m_init_layer_threshold
    
    private :: m_forward_propagation
    private :: m_backward_propagation
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
    !* (2). �����ʼ��Ȩֵ����ֵ;
    !* (3). ��ʼ�������.
    subroutine m_init( this, l_count, l_node_count, loss_fun )
    implicit none
        class(NNStructure), intent(inout) :: this
        integer, intent(in) :: l_count
        integer, dimension(:), intent(in) :: l_node_count 
        class(BaseLossFunction), target, optional, intent(in) :: loss_fun

        if( .not. this % is_init ) then
            
            this % layers_count = l_count
			
			call this % allocate_pointer()
			
            this % layers_node_count = l_node_count
			
			call this % allocate_memory()

            this % is_allocate_done = .true.
            
            call this % init_layer_weight()
            call this % init_layer_threshold()
            
            if (PRESENT(loss_fun)) then
                call this % set_loss_function(loss_fun)
            end if
            
            this % is_init = .true.
            
            call LogDebug("NNStructure: SUBROUTINE m_init")
        end if

        return
    end subroutine m_init
    !====
    
    !* �Ƿ��ʼ��
    subroutine c_isInit( this, init )
    implicit none
        class(NNStructure), intent(inout)  :: this
        logical,            intent(out)    :: init

        if (this % is_allocate_done    .and. this % is_init_act_fun .and. &
            this % is_init_input_layer .and. this % is_init_output_layer) then
            this % is_init = .true.
        end if
        
        init = this % is_init

        return
    end subroutine c_isInit
    !==== 
    
    
    !* ��ʼ��ָ����ļ����
    subroutine m_init_activation_function( this, layer_index, act_fun )
    implicit none
        class(NNStructure), intent(inout) :: this
        integer, intent(in) :: layer_index
        class(BaseActivationFunction), target, intent(in) :: act_fun
    
        this % pt_Layer(layer_index) % act_fun => act_fun
        
        call LogDebug("NNStructure: SUBROUTINE m_init_activation_function")
        
        return
    end subroutine m_init_activation_function
    !====
    
    !* ���ü����
    subroutine m_set_loss_function( this, loss_fun )
    implicit none
        class(NNStructure), intent(inout) :: this
        class(BaseLossFunction), target, intent(in) :: loss_fun
    
        this % loss_function => loss_fun
        
        this % is_init_loss_fun = .true.
        
        call LogDebug("NNStructure: SUBROUTINE m_set_loss_function")
        
        return
    end subroutine m_set_loss_function
    !====
    
    !* ��ʼ�������
    subroutine m_init_input_layer( this, X_in )
    implicit none
        class(NNStructure), intent(inout) :: this
        real(PRECISION), dimension(:), intent(in) :: X_in 
    
        this % X = X_in
        this % pt_Layer( 0 ) % Z = X_in
        this % is_init_input_layer = .true.
        
        call LogDebug("NNStructure: SUBROUTINE m_init_input_layer")
        
        return
    end subroutine m_init_input_layer
    !====
    
    !* ��ʼ�������
    subroutine m_init_output_layer( this, t )
    implicit none
        class(NNStructure), intent(inout) :: this
        real(PRECISION), dimension(:), intent(in) :: t 
        
        this % t = t
        this % is_init_output_layer = .true.
        
        call LogDebug("NNStructure: SUBROUTINE m_init_output_layer")
        
        return
    end subroutine m_init_output_layer
    !====
    
    !* �����ʼ����ֵ��Ĭ�ϳ�ʼ����(-1,1)
    !* ��Train�����У������������ó�ʼ��.
    subroutine m_init_layer_weight( this )
    implicit none
        class(NNStructure), intent(inout) :: this
     
        integer :: layer_index, l_count
        
        l_count = this % layers_count
        
        call RANDOM_SEED()
        
        call LogInfo("NNStructure: SUBROUTINE m_init_layer_weight")
        
        do layer_index=1, l_count
            associate (                            &              
                W => this % pt_W(layer_index) % W  &
            )
                
            call RANDOM_NUMBER(W)
            W = 2.0 * W - 1.0
            
            end associate
        end do
        
        this % is_init_weight = .true.
        
        call LogDebug("NNStructure: SUBROUTINE m_init_layer_weight")
        
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
        
        call LogInfo("NNStructure: SUBROUTINE m_init_layer_threshold")
        
        do layer_index=1, l_count
            associate (                                        &              
                Theta => this % pt_Theta(layer_index) % Theta  &
            )
                
            call RANDOM_NUMBER(Theta) 
            Theta = 2.0 * Theta - 1.0
            
            end associate
        end do
  
        this % is_init_threshold = .true.
        
        call LogDebug("NNStructure: SUBROUTINE m_init_layer_threshold")
        
        return
    end subroutine m_init_layer_threshold
    !====
    
    !* ǰ����㣬��������ֵ����������������ֵ��
    !* ������Ԥ��ֵ
    subroutine m_forward_propagation( this, X, t, y )
    implicit none
        class(NNStructure), intent(inout) :: this
        !* X ������ֵ��t ��ʵ�������y ������Ԥ�����
        real(PRECISION), dimension(:), intent(in) :: X
        real(PRECISION), dimension(:), intent(in) :: t
        real(PRECISION), dimension(:), intent(inout) :: y
        
        integer :: l_count
        
        l_count = this % layers_count
        
        call this % init_input_layer( X )
        call this % init_output_layer( t )
            
        !* ǰ����㣺�������в��еľֲ����� S��R��Z
        call this % get_all_layer_local_var()
        
        y = this % pt_Layer(l_count) % Z 
        
        call LogDebug("NNStructure: SUBROUTINE m_forward_propagation")
        
        return
    end subroutine m_forward_propagation
    !====
    

      
    !* ������㣬�������������������ĵ���
    subroutine m_backward_propagation( this, X, t, y )
    implicit none
        class(NNStructure), intent(inout) :: this
        !* X ������ֵ��t ��ʵ�������y ������Ԥ�����
        real(PRECISION), dimension(:), intent(in) :: X
        real(PRECISION), dimension(:), intent(in) :: t
        real(PRECISION), dimension(:), intent(inout) :: y
        
        integer :: M, N, layer_index, l_count
        
        l_count = this % layers_count
        
        call this % init_input_layer( X )
        call this % init_output_layer( t )
        
        !* ǰ����㣺�������в��еľֲ����� S��R��Z
        call this % get_all_layer_local_var()
        
        !* ������㣺���е�����Ϣ
        call this % get_all_derivative_variable()
        
        !* �Ե�����Ϣ������ͣ��ۼ�BP�㷨��Ҫ�õ�.
        do layer_index=1, l_count        
            associate (                                                   &
                dW         => this % pt_Layer( layer_index ) % dW,        &
                dTheta     => this % pt_Layer( layer_index ) % dTheta,    &
                sum_dW     => this % pt_Layer( layer_index ) % sum_dW,    &               
                sum_dTheta => this % pt_Layer( layer_index ) % sum_dTheta &
            )
        
            sum_dW = sum_dW + dW   
            sum_dTheta = sum_dTheta + dTheta
            
            end associate       
        end do
        
        y = this % pt_Layer(l_count) % Z 
        
        call LogDebug("NNStructure: SUBROUTINE m_backward_propagation")
        
        return
    end subroutine m_backward_propagation
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
            associate (                                        &              
                W     => this % pt_W(layer_index) % W,         &
                Theta => this % pt_Theta(layer_index) % Theta, &
                S     => this % pt_Layer(layer_index) % S,     &
                R     => this % pt_Layer(layer_index) % R,     &
                Z     => this % pt_Layer(layer_index) % Z,     &
                Z1    => this % pt_Layer(layer_index-1) % Z    &
            )
        
            !* S^{k} = W^{k}*Z^{k-1}
            S = MATMUL(W, Z1) 
            
            !* R^{k} = S^{k} - Theta^{k}
            R = S - Theta
                
            !* Z^{k} = f(R^{k})
            !call this % my_act_fun % f_vect( R, Z )
            call this % pt_Layer(layer_index) % act_fun % f_vect( R, Z )      
            
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
                      
        !* zeta��zn�ĵ������� zn - t
        associate (                                                    &              
            d_Matrix_part => this % pt_Layer(l_count) % d_Matrix_part, &
            Z             => this % pt_Layer(l_count) % Z,             &
            t             => this % t                                  &
        )
        
        !d_Matrix_part = Z - t
        call this % loss_function % df(t, Z, d_Matrix_part)
            
        end associate
        
        do layer_index=l_count-1, 1, -1
            associate (                                                             &              
                M => this % layers_node_count(layer_index),                         &
                N => this % layers_node_count(layer_index + 1),                     &
                R => this % pt_Layer(layer_index + 1) % R,                          &
                W => this % pt_W(layer_index + 1) % W,                              &
                matrix_part_next => this % pt_Layer(layer_index+1) % d_Matrix_part, &
                matrix_part => this % pt_Layer(layer_index) % d_Matrix_part         &
            )
            
            allocate( WT_GammaT(M, N) )
                
            do j=1, N
                call this % pt_Layer(layer_index+1) % act_fun % df( j, R, df_to_dr ) 
                WT_GammaT(:, j) = W(j, :) * df_to_dr
            end do
            
            matrix_part = MATMUL(WT_GammaT, matrix_part_next)
            
            deallocate( WT_GammaT )
            
            end associate    
        end do
        
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

        associate (                                                      &                              
            dW          => this % pt_Layer(layer_index) % dW,            &
            R           => this % pt_Layer(layer_index) % R,             &
            matrix_part => this % pt_Layer(layer_index) % d_Matrix_part, &
            Z1          => this % pt_Layer(layer_index-1) % Z            &          
        )
        
        do i=1, N     
            call this % pt_Layer(layer_index) % act_fun % df( i, R, df_to_dr )
            
            !* dW^{k}_{ij} = f'(r^{k}_i) * z^{k-1}_j * E_i * d_Matrix_part
            do j=1, M
                dW(i, j) = df_to_dr * Z1(j) * matrix_part(i)
            end do
        end do
        
        end associate
        
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

        associate (                                                     &                              
            dTheta      => this % pt_Layer(layer_index) % dTheta,       &
            R           => this % pt_Layer(layer_index) % R,            &
            matrix_part => this % pt_Layer(layer_index) % d_Matrix_part &
        )
        
        do i=1, N          
            call this % pt_Layer(layer_index) % act_fun % df( i, R, df_to_dr )
        
            !* dTheta_{i} = -f'(r_i) * E_i * d_Matrix_part
            dTheta(i) = -df_to_dr * matrix_part(i)
        end do
        
        end associate
        
        call LogDebug("NNStructure: SUBROUTINE m_get_layer_dTheta")
        
        return
    end subroutine m_get_layer_dTheta    
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
            allocate( this % pt_Layer( layer_index ) % sum_dW(N,M) )
            
            allocate( this % pt_Layer( layer_index ) % dTheta(N) )
            allocate( this % pt_Layer( layer_index ) % sum_dTheta(N) )
        end do
        
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
        
		do layer_index=1, this%layers_count
			
			deallocate( this % pt_W( layer_index ) % W )
			deallocate( this % pt_Theta( layer_index ) % Theta )
			deallocate( this % pt_Layer( layer_index ) % S )
			deallocate( this % pt_Layer( layer_index ) % R )
			deallocate( this % pt_Layer( layer_index ) % Z )
            deallocate( this % pt_Layer( layer_index ) % dW )
			deallocate( this % pt_Layer( layer_index ) % dTheta )
            deallocate( this % pt_Layer( layer_index ) % sum_dW )
			deallocate( this % pt_Layer( layer_index ) % sum_dTheta )
			
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