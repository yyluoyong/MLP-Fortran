!* ��ģ�鶨��������������õ�������ṹ��
!* ����Ĳ�������μ�PDF�ĵ���
module mod_NNParameter
use mod_Precision
use mod_BaseActivationFunction
implicit none
    !---------------------------------------------------------
    ! �����֮���Ȩ��
    !---------------------------------------------------------
    type :: Layer_Weight
        !* ע�⣺W��������Ľ����Ŀ�ֱ�Ϊ M,N
        !*       ��WΪ N��M �ľ���.
        real(kind=PRECISION), dimension(:,:), allocatable :: W    
    end type
    !=========================================================

    
    
    !---------------------------------------------------------
    ! �����ֵ
    !---------------------------------------------------------
    type :: Layer_Threshold
        !* ����Ĵ�С�Ǹ���ֵ��Ӧ��Ľڵ���Ŀ
        real(kind=PRECISION), dimension(:), allocatable :: Theta     
    end type
    !=========================================================

    
    
    !---------------------------------------------------------
    ! �����õ��ľֲ����飺���롢�����
    !---------------------------------------------------------
    type :: Layer_Local_Array
        !* ����Ĵ�С�Ǹ���ֵ��Ӧ��Ľڵ���Ŀ
        !* S���������飬R=S-theta��Z��������飬Z=f(R)=f(S-Theta)
        real(kind=PRECISION), dimension(:), allocatable :: S
        real(kind=PRECISION), dimension(:), allocatable :: R
        real(kind=PRECISION), dimension(:), allocatable :: Z
        
        !* (Gamma^{k+1} W^{k+1})^T ... (Gamma^{n} W^{n}) p_zeta/p_zn
        real(kind=PRECISION), dimension(:), allocatable :: d_Matrix_part
        
        !* ����zeta��ʾ������
        !* zeta��W�ĵ���
        !* �����С��д�С�ֱ��Ǹ�Ȩ��W���ӵ�����Ľڵ���Ŀ
        real(kind=PRECISION), dimension(:,:), allocatable :: dW
        
        !* ��������zeta��W�ĵ�������ƽ��
        real(kind=PRECISION), dimension(:,:), allocatable :: avg_dW
        
        !* zeta��Theta�ĵ���
        !* ����Ĵ�С�Ǹ���ֵ��Ӧ��Ľڵ���Ŀ
        real(kind=PRECISION), dimension(:), allocatable :: dTheta   
        
        !* ��������zeta��Theta�ĵ�������ƽ��
        real(kind=PRECISION), dimension(:), allocatable :: avg_dTheta  
        
        !* �����
        !* ע��BaseActivationFunction �ǳ����࣬����ʹ�ö�̬����.
        class(BaseActivationFunction), pointer :: act_fun
    end type 
    !=========================================================
    
    
    !* W��Theta ����Ҳ����ͳһ�ŵ� Layer_Local_Array �ṹ���У�
    !* ���ﵥ��������Ϊ��Ӧ��δ��������ܵ��޸ġ�
    
end module